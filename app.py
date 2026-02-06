import streamlit as st
from datetime import datetime
from typing import Dict, List
import pandas as pd
import json
import hashlib
from data_loader import fetch_reviews
from analysis import (
    load_sentiment_model,
    load_emotion_model,
    analyze_texts,
    analyze_aspects,
)
from visualization import (
    build_plotly_sentiment_bar,
    build_plotly_emotion_bar,
    build_sentiment_trend,
    build_wordcloud_figure,
    build_nps_gauge,
    build_steam_vs_model_comparison,
    build_aspect_bar_from_dict,
)
from ui_components import (
    render_sidebar_controls,
    prepare_df_for_table,
    render_filters_and_review_list,
)
from spam_filter import filter_spam_reviews, filter_curse_words_from_text

st.set_page_config(page_title="Analiza sentymentu i emocji w recenzjach gier", layout="wide")
st.title("Analiza sentymentu i emocji w recenzjach gier")

# Definicje nazw modeli
SENTIMENT_MODEL_NAME = "clapAI/roberta-large-multilingual-sentiment"
EMOTION_MODEL_PL_NAME = "visegradmedia-emotion/Emotion_RoBERTa_pooled_V4"
EMOTION_MODEL_EN_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Funkcja służąca do automatycznego generowania nazw dla plików CSV/JSON
def _experiment_meta(config: Dict) -> Dict:
    cfg_str = json.dumps(config, sort_keys=True)
    h = hashlib.sha256(cfg_str.encode()).hexdigest()[:12]
    return {"run_id": h, "config_hash": h, "timestamp": datetime.now().isoformat()}

def calculate_nps(reviews: List[Dict]) -> Dict[str, int]:
    """
    Net Promoter Score (NPS).
    NPS = (Promoters% - Detractors%) * 100
    Promoters: sentiment_score > 0.7 (ekwiwalent 7/10) lub emocja "radość"
    Detractors: sentiment_score < 0.5 (ekwiwalent 5/10) lub emocja "złość"/"smutek"
    Passives: pozostałe (nie są uwzględniane w obliczeniach)
    """
    promoters_set = set()
    detractors_set = set()
    
    # Każda recenzja może być tylko w jednej kategorii - priorytet: detractors > promoters > passives
    for r in reviews:
        review_id = id(r)  # Użyj ID obiektu jako unikalnego identyfikatora
        sentiment_score = r.get("sentyment_score", 0.5)
        emotion_label = r.get("emocja_label", "")
        
        # Sprawdź najpierw detractors (wyższy priorytet)
        if sentiment_score < 0.5 or emotion_label in ["złość", "smutek"]:
            detractors_set.add(review_id)
        # Jeśli nie jest detractor, sprawdź promoters
        elif sentiment_score > 0.7 or emotion_label == "radość":
            promoters_set.add(review_id)
    
    promoters = len(promoters_set)
    detractors = len(detractors_set)
    total = len(reviews)
    # Standardowa formuła NPS: (promoters - detractors) / total * 100
    nps_score = ((promoters - detractors) / total * 100) if total else 0
    passives = max(0, total - promoters - detractors)  # Upewnij się, że nie jest ujemne
    return {
        "nps_score": round(nps_score, 1), 
        "promoters": promoters, 
        "detractors": detractors, 
        "passives": passives,
        "total": total
    }

settings: Dict = render_sidebar_controls()

# Klucz unikalny dla sesji - automatycznie generowana nazwa
emotion_model_id = "visegrad" if settings.get("selected_language") == "polish" else "hartmann"
emotion_model_name = EMOTION_MODEL_PL_NAME if settings.get("selected_language") == "polish" else EMOTION_MODEL_EN_NAME

current_run_key = (
    settings.get("app_id"),
    settings.get("num_reviews"),
    settings.get("selected_language"),
    f"sentiment-clapAI-roberta-emotion-{emotion_model_id}"
)

if "analysis_results" in st.session_state and st.session_state.get("last_run_key") == current_run_key:
    results = st.session_state["analysis_results"]
    merged = results["merged"]
    recenzje_spam = results.get("spam_reviews", [])
    tekst_sent = results["tekst_sent"]
    tekst_emot = results["tekst_emot"]
    labels_sent = results["labels_sent"]
    labels_emot = results["labels_emot"]
    aspects = results["aspects"]
    nps = results["nps"]
    meta = results["meta"]
elif settings.get("run_button") and settings.get("app_id"):
    start_time = datetime.now()

    with st.spinner(f"Pobieranie recenzji, wczytuję modele: {SENTIMENT_MODEL_NAME} oraz {emotion_model_name}"):
        try:
            # Ładujemy model sentymentu (bezpiecznie)
            tok_sent, model_sent, id2label_sent, labels_map_sent, dev_sent = load_sentiment_model()

            # Ładujemy model emocji
            tok_emot, model_emot, labels_emot_list, dev_emot, label_mapping_emot = load_emotion_model(settings["selected_language"])

            # Pobieramy recenzje - mierzymy czas pobierania osobno
            download_start = datetime.now()
            recenzje_raw = fetch_reviews(settings["app_id"], settings["num_reviews"], settings["selected_language"])
            download_duration = (datetime.now() - download_start).total_seconds()

            if not recenzje_raw:
                st.error("Brak recenzji – sprawdź AppID i limity API.")
                st.stop()

            # Filtruj spam przed analizą
            recenzje_normalne, recenzje_spam = filter_spam_reviews(recenzje_raw, settings["selected_language"])
            
            if recenzje_spam:
                st.warning(f"Znaleziono {len(recenzje_spam)} recenzji oznaczonych jako spam/niskiej jakości. Zostaną one wyświetlone w osobnej tabeli.")
            
            if not recenzje_normalne:
                st.error("Wszystkie recenzje zostały odfiltrowane jako spam. Spróbuj pobrać więcej recenzji.")
                st.stop()

            # Analiza sentymentu (tylko normalne recenzje)
            wyniki_sent, tekst_sent = analyze_texts(recenzje_normalne, tok_sent, model_sent, id2label_sent, dev_sent, labels_map_sent)
            # Analiza emocji (z mapowaniem etykiet) - tylko normalne recenzje
            wyniki_emot, tekst_emot = analyze_texts(recenzje_normalne, tok_emot, model_emot, labels_emot_list, dev_emot, label_mapping_emot)

            # Scalanie wyników - dopasowanie po timestamp, aby uniknąć utraty recenzji
            # Tworzymy słowniki indeksowane po timestamp dla szybszego wyszukiwania
            # Używamy listy zamiast dict dla recenzji z tym samym timestampem
            sent_dict = {}
            for s in wyniki_sent:
                ts = s.get("timestamp")
                if ts is not None:
                    if ts not in sent_dict:
                        sent_dict[ts] = []
                    sent_dict[ts].append(s)
            
            emot_dict = {}
            for e in wyniki_emot:
                ts = e.get("timestamp")
                if ts is not None:
                    if ts not in emot_dict:
                        emot_dict[ts] = []
                    emot_dict[ts].append(e)
            
            # Używamy wszystkich unikalnych timestampów z obu list
            all_timestamps = set(sent_dict.keys()) | set(emot_dict.keys())
            
            # Ostrzeżenie jeśli liczby się nie zgadzają
            if len(wyniki_sent) != len(wyniki_emot):
                st.warning(f"Różnica w liczbie wyników: sentyment={len(wyniki_sent)}, emocje={len(wyniki_emot)}. "
                          f"Możliwe, że niektóre recenzje zostały pominięte podczas analizy.")
            
            merged = []
            for ts in sorted(all_timestamps):
                sent_list = sent_dict.get(ts, [])
                emot_list = emot_dict.get(ts, [])
                
                # Dopasowujemy wyniki - jeśli jest więcej wyników dla jednego timestampu, 
                # dopasowujemy po tekście lub bierzemy pierwszy dostępny
                max_pairs = max(len(sent_list), len(emot_list))
                
                for i in range(max_pairs):
                    s = sent_list[i] if i < len(sent_list) else None
                    e = emot_list[i] if i < len(emot_list) else None
                    
                    # Jeśli brakuje jednego z wyników, użyj domyślnych wartości
                    if s is None and e is None:
                        continue  # Pomiń jeśli oba są None
                    if s is None:
                        s = {"tekst": e.get("tekst", ""), "etykieta_modelu": "neutralny", 
                             "sentyment_score": 0.5, "polarity": "neutralny",
                             "helpful": e.get("helpful", 0), "purchase_type": e.get("purchase_type", ""),
                             "voted_up": e.get("voted_up", False), "author": e.get("author", {}),
                             "weighted_vote_score": e.get("weighted_vote_score", 0.0),
                             "comment_count": e.get("comment_count", 0),
                             "early_access": e.get("early_access", False),
                             "received_for_free": e.get("received_for_free", False)}
                    if e is None:
                        e = {"etykieta_modelu": "neutralny", "pewnosc": 0.5, "polarity": "neutralny"}
                    
                    merged.append({
                        "tekst": s.get("tekst", ""),
                        "sentyment_label": s.get("etykieta_modelu", "neutralny"),
                        "sentyment_score": s.get("sentyment_score", 0.5),
                        "emocja_label": e.get("etykieta_modelu", "neutralny"),
                        "emocja_score": e.get("pewnosc", 0.5),
                        "timestamp": ts,
                        "sentyment_polar": s.get("polarity", "neutralny"),
                        "emocja_polar": e.get("polarity", "neutralny"),
                        "helpful": s.get("helpful", 0),
                        "purchase_type": s.get("purchase_type", ""),
                        "voted_up": s.get("voted_up", False),
                        "author": s.get("author", {}),
                        "weighted_vote_score": s.get("weighted_vote_score", 0.0),
                        "comment_count": s.get("comment_count", 0),
                        "early_access": s.get("early_access", False),
                        "received_for_free": s.get("received_for_free", False),
                        "language": settings["selected_language"]
                    })

            # Analiza aspektów i NPS
            aspects = analyze_aspects(merged, language=settings["selected_language"])
            nps = calculate_nps(merged)
            meta = _experiment_meta(settings)

            # Zapisz w sesji (włączając spam dla późniejszego wyświetlenia)
            st.session_state["analysis_results"] = {
                "merged": merged,
                "spam_reviews": recenzje_spam,
                "tekst_sent": tekst_sent,
                "tekst_emot": tekst_emot,
                "labels_sent": id2label_sent,
                "labels_emot": labels_emot_list,
                "aspects": aspects,
                "nps": nps,
                "meta": meta,
            }
            st.session_state["last_run_key"] = current_run_key
        except Exception as e:
            st.error(f"Wystąpił błąd podczas analizy: {e}")
            st.stop()

    # Oblicz czas operacji i wyświetl info
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    lang_display = "polskich" if settings['selected_language'] == 'polish' else "angielskich"
    st.success(f"Pobrano {len(recenzje_raw)} {lang_display} recenzji w {download_duration:.2f} sekund (całkowity czas analizy: {total_duration:.2f} sekund)")
    st.header(f"Analiza gry: {settings['selected_game_name']}")

# ------------- WIZUALIZACJA -------------
if "merged" in locals():
    df_table = prepare_df_for_table(merged)
    st.markdown("---")

    # Główne wykresy
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(build_plotly_sentiment_bar(df_table), use_container_width=True, key="sentiment_bar")
    with col2:
        st.plotly_chart(build_plotly_emotion_bar(df_table), use_container_width=True, key="emotion_bar")
    
    # Wykres Steam vs Model
    st.markdown("### Porównanie: Ocena Steam vs. Sentyment modelu")
    st.info("""
    - **Steam Pozytywna**: Recenzje, które użytkownicy Steam ocenili jako pozytywne (voted_up=True)
    - **Steam Negatywna**: Recenzje, które użytkownicy Steam ocenili jako negatywne (voted_up=False)
    
    Każdy słupek jest podzielony na 3 kolory pokazujące, jak model sentymentu sklasyfikował te recenzje:
    - **Pozytywny** (zielony): Model uznał recenzję za pozytywną
    - **Neutralny** (niebieski): Model uznał recenzję za neutralną  
    - **Negatywny** (czerwony): Model uznał recenzję za negatywną
    """)
    st.plotly_chart(build_steam_vs_model_comparison(df_table), use_container_width=True, key="steam_vs_model")

    # Trend sentymentu w czasie
    st.markdown("### Trend sentymentu w czasie")
    st.info("Pokazuje, jak zmieniał się sentyment w czasie")
    st.plotly_chart(build_sentiment_trend(df_table), use_container_width=True, key="sentiment_trend")

    # Wykres aspektów (jeśli dostępne)
    if 'aspects' in locals() and aspects:
        st.markdown("### Średni sentyment per aspekt (gaming)")
        st.plotly_chart(build_aspect_bar_from_dict(aspects), use_container_width=True, key="aspect_bar")
        
        # Wyświetl przykładowe recenzje dla każdego aspektu
        st.markdown("#### Przykładowe recenzje per aspekt")
        aspect_names = {
            "gameplay": "Gameplay",
            "graphics": "Grafika",
            "performance": "Wydajność",
            "story": "Fabuła",
            "multiplayer": "Multiplayer",
            "audio": "Audio",
            "ui": "Interfejs",
            "monetization": "Monetyzacja"
        }
        
        # Sortuj aspekty według liczby wzmianek (malejąco)
        sorted_aspects = sorted(aspects.items(), key=lambda x: x[1].get("mention_count", 0), reverse=True)
        
        for aspect_key, aspect_data in sorted_aspects:
            if aspect_data.get("mention_count", 0) > 0:
                aspect_display_name = aspect_names.get(aspect_key, aspect_key.capitalize())
                sample_reviews = aspect_data.get("sample_reviews", [])
                avg_sentiment = aspect_data.get("avg_sentiment", 0.5)
                mention_count = aspect_data.get("mention_count", 0)
                
                with st.expander(f"{aspect_display_name} (wzmianek: {mention_count}, średni sentyment: {avg_sentiment:.2f})"):
                    if sample_reviews:
                        for i, review in enumerate(sample_reviews[:3], 1):
                            review_text = review.get("tekst") or review.get("review") or ""
                            sentiment_label = review.get("sentyment_label", "N/A")
                            sentiment_score = review.get("sentyment_score", 0.5)
                            author_info = review.get("author", {})
                            user_id = author_info.get("steamid", "N/A") if isinstance(author_info, dict) else "N/A"
                            
                            st.markdown(f"**Przykład {i}:** (User ID: {user_id})")
                            st.text_area("Recenzja", review_text[:300] + ("..." if len(review_text) > 300 else ""), 
                                       height=80, key=f"aspect_{aspect_key}_review_{i}", disabled=True, label_visibility="collapsed")
                            st.caption(f"Sentyment: {sentiment_label} (score: {sentiment_score:.2f})")
                            st.markdown("---")
                    else:
                        st.info("Brak przykładowych recenzji dla tego aspektu.")

    # NPS
    if nps:
        st.markdown("### Net Promoter Score – lojalność graczy")
        st.info("NPS to metryka lojalności: -100 (sami wrogowie) do +100 (sami fani). Promoters: sentyment > 0.7 lub radość. Detractors: sentyment < 0.5 lub złość/smutek.")
        st.plotly_chart(build_nps_gauge(nps["nps_score"]), use_container_width=True, key="nps_gauge")
        col_nps1, col_nps2, col_nps3 = st.columns(3)
        with col_nps1:
            st.metric("Promoters (entuzjaści)", nps["promoters"])
        with col_nps2:
            st.metric("Detractors (krytycy)", nps["detractors"])
        with col_nps3:
            st.metric("Passives (neutralni)", nps.get("passives", 0))

    # Chmura słów (bez wulgaryzmów)
    st.markdown("### Chmura słów – całość recenzji")
    # Filtruj wulgaryzmy z tekstu przed tworzeniem word cloud
    # Użyj języka z merged reviews lub domyślnego
    review_language = merged[0].get("language", "english") if merged else "english"
    clean_text_sent = filter_curse_words_from_text(tekst_sent or "", review_language)
    clean_text_emot = filter_curse_words_from_text(tekst_emot or "", review_language)
    wc_fig = build_wordcloud_figure(clean_text_sent + " " + clean_text_emot)
    st.pyplot(wc_fig)
    
    # Tabela spam recenzji
    if 'recenzje_spam' in locals() and recenzje_spam:
        st.markdown("### Recenzje oznaczone jako spam/niskiej jakości")
        st.info(f"Znaleziono {len(recenzje_spam)} recenzji, które zostały odfiltrowane jako spam (zbyt krótkie, tylko wulgaryzmy, losowe znaki).")
        spam_df = pd.DataFrame(recenzje_spam)
        if not spam_df.empty:
            # Przygotuj kolumny do wyświetlenia
            spam_display = spam_df.copy()
            if "review" in spam_display.columns:
                spam_display["tekst"] = spam_display["review"]
            if "author" in spam_display.columns:
                spam_display["author_display"] = spam_display["author"].apply(
                    lambda x: x.get("steamid", "N/A") if isinstance(x, dict) else str(x)
                )
            if "timestamp" in spam_display.columns:
                spam_display["date"] = pd.to_datetime(spam_display["timestamp"], errors='coerce', unit='s').dt.strftime("%Y-%m-%d")
            
            display_cols = []
            if "date" in spam_display.columns:
                display_cols.append("date")
            if "author_display" in spam_display.columns:
                display_cols.append("author_display")
            if "tekst" in spam_display.columns:
                display_cols.append("tekst")
            
            if display_cols:
                st.dataframe(
                    spam_display[display_cols].rename(columns={
                        "tekst": "recenzja",
                        "author_display": "user_id"
                    }),
                    height=300,
                    use_container_width=True,
                    hide_index=True
                )

    render_filters_and_review_list(merged)
    
    # Eksport (po tabeli recenzji, zawiera również spam)
    st.markdown("---")
    st.markdown("### Eksport danych")
    col_csv, col_json = st.columns(2)
    
    # Przygotuj dane do eksportu - włącz spam recenzje
    all_reviews_for_export = merged.copy()
    if 'recenzje_spam' in locals() and recenzje_spam:
        # Dodaj flagę is_spam do każdej recenzji
        for review in all_reviews_for_export:
            review["is_spam"] = False
        for spam_review in recenzje_spam:
            spam_review_copy = spam_review.copy()
            spam_review_copy["is_spam"] = True
            # Dodaj brakujące pola dla zgodności
            if "sentyment_label" not in spam_review_copy:
                spam_review_copy["sentyment_label"] = "N/A"
            if "sentyment_score" not in spam_review_copy:
                spam_review_copy["sentyment_score"] = 0.5
            if "emocja_label" not in spam_review_copy:
                spam_review_copy["emocja_label"] = "N/A"
            if "emocja_score" not in spam_review_copy:
                spam_review_copy["emocja_score"] = 0.5
            all_reviews_for_export.append(spam_review_copy)
    
    with col_csv:
        csv = pd.DataFrame(all_reviews_for_export).to_csv(index=False).encode("utf-8")
        st.download_button("Pobierz CSV", data=csv, file_name=f"steam_{meta['run_id']}.csv", mime="text/csv", use_container_width=True)

    with col_json:
        json_data = {
            "metadata": meta, 
            "nps": nps, 
            "aspects": aspects, 
            "reviews": all_reviews_for_export,
            "spam_count": len(recenzje_spam) if 'recenzje_spam' in locals() else 0
        }
        st.download_button("Pobierz JSON", data=json.dumps(json_data, ensure_ascii=False, indent=2),
                           file_name=f"steam_{meta['run_id']}.json", mime="application/json", use_container_width=True)
else:
    st.info("Wprowadź tytuł gry, ustaw parametry i kliknij 'Analizuj recenzje (sentyment + emocje)', aby przeprowadzić analizę.")
