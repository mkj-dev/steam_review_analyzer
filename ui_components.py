from typing import Dict, List
import streamlit as st
import pandas as pd
from data_loader import search_game_by_title
from datetime import datetime
from analysis import summarize_reviews

def render_sidebar_controls() -> Dict:
    st.sidebar.header("Ustawienia analizy")
    title_query = st.sidebar.text_input("Tytuł gry:", value="Cyberpunk")
    search_results = search_game_by_title(title_query) if title_query else []

    selected_game_name, app_id = "", None
    if search_results:
        game_options = {f"{g['name']} (AppID: {g['appid']})": g['appid'] for g in search_results[:10]}
        selected_label = st.sidebar.selectbox("Wybierz grę:", list(game_options.keys()))
        app_id = game_options[selected_label]
        selected_game_name = selected_label.split(" (")[0]
    else:
        st.sidebar.write("Brak wyników wyszukiwania. Wpisz tytuł i poczekaj chwilę.")

    num_reviews = st.sidebar.slider("Liczba recenzji:", 10, 1000, 200, step=10)
    language_choice = st.sidebar.radio("Język recenzji:", ["Angielski", "Polski"], index=0)
    language_map = {"Angielski": "english", "Polski": "polish"}
    selected_language = language_map[language_choice]
    run_button = st.sidebar.button("Analizuj recenzje (sentyment + emocje)")

    return {
        "title_query": title_query,
        "selected_game_name": selected_game_name,
        "app_id": app_id,
        "num_reviews": num_reviews,
        "selected_language": selected_language,
        "run_button": run_button,
        "language_choice": language_choice,
    }

def format_author_info(author_data):
    """Wspólna funkcja formatująca informacje o autorze."""
    if not isinstance(author_data, dict):
        return str(author_data)
    steamid = author_data.get("steamid", "N/A")
    num_games = author_data.get("num_games_owned", 0)
    num_reviews = author_data.get("num_reviews", 0)
    playtime = author_data.get("playtime_at_review_hours", 0)
    return f"ID:{steamid} | G:{num_games} | R:{num_reviews} | {playtime}h"

def prepare_df_for_table(merged_list: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(merged_list)
    expected = ["tekst", "sentyment_label", "sentyment_score", "emocja_label", "emocja_score",
                "timestamp", "sentyment_polar", "emocja_polar", "author", "weighted_vote_score",
                "comment_count", "early_access", "received_for_free", "purchase_type", "voted_up"]

    for col in expected:
        if col not in df.columns:
            df[col] = None

    # Parsowanie timestamp -> date
    def ts_to_date(v):
        try:
            if v and int(v) > 0:
                return datetime.fromtimestamp(int(v)).date()
        except Exception:
            return None
        return None

    df["date"] = df["timestamp"].apply(ts_to_date)

    # Przygotowanie kolumny autora do wyświetlenia
    if "author" in df.columns:
        df["author_display"] = df["author"].apply(format_author_info)

    return df

def render_filters_and_review_list(merged: List[Dict]):
    st.subheader("Filtry recenzji")

    # Wygeneruj źródłowy DataFrame
    if not merged:
        st.info("Brak danych do wyświetlenia.")
        return
    df_all = pd.DataFrame(merged)

    # Podsumowanie fraz (opcjonalnie)
    try:
        lang = df_all.iloc[0].get("language", "english") if not df_all.empty else "english"
        summary = summarize_reviews(merged, top_n=5, lang=lang)
        if summary.get("positive") or summary.get("negative"):
            st.markdown("### Szybkie podsumowanie (top fraz)")
            colp, coln = st.columns(2)
            with colp:
                st.write("Top fraz – recenzje pozytywne")
                for ph, cnt in summary.get("positive", []):
                    st.write(f"- {ph} ({cnt})")
            with coln:
                st.write("Top fraz – recenzje negatywne")
                for ph, cnt in summary.get("negative", []):
                    st.write(f"- {ph} ({cnt})")
    except Exception:
        pass

    # Przygotowanie listy wartości filtrów
    all_sentiments = sorted(df_all["sentyment_polar"].dropna().unique().tolist())
    all_emotions = sorted(df_all["emocja_label"].dropna().unique().tolist())
    max_helpful = int(df_all["helpful"].fillna(0).astype(int).max()) if "helpful" in df_all.columns else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        sent_choice = st.selectbox("Filtruj po sentymencie:", ["Wszystkie"] + all_sentiments)
    with col2:
        emot_choice = st.selectbox("Filtruj po emocji:", ["Wszystkie"] + all_emotions)
    with col3:
        helpful_min = st.slider("Min. liczba pomocnych głosów:", 0, max_helpful, 0)

    keyword_input = st.text_input("Wyszukaj słowo kluczowe:", "")

    # Filtrowanie DataFrame używając pandas
    df_filtered = df_all.copy()

    if sent_choice != "Wszystkie":
        df_filtered = df_filtered[df_filtered["sentyment_polar"].str.lower() == sent_choice.lower()]

    if emot_choice != "Wszystkie":
        df_filtered = df_filtered[df_filtered["emocja_label"] == emot_choice]

    if helpful_min:
        df_filtered = df_filtered[df_filtered["helpful"].fillna(0).astype(int) >= helpful_min]

    kw = keyword_input.strip().lower()
    if kw:
        df_filtered = df_filtered[df_filtered["tekst"].fillna("").str.lower().str.contains(kw, na=False)]

    total = len(df_filtered)
    # Paginacja
    page_size = 50
    if total > page_size:
        page = st.number_input("Strona", 1, max(1, (total - 1) // page_size + 1), 1)
        start_idx = (page - 1) * page_size
        page_items = df_filtered.iloc[start_idx:start_idx + page_size]
    else:
        page_items = df_filtered

    st.markdown(f"**Znaleziono {total} recenzji**")

    if page_items.empty:
        st.info("Brak recenzji do wyświetlenia.")
        return

    df_display = page_items.copy()
    # Parsowanie daty do stringa dla wyświetlenia
    if "timestamp" in df_display.columns:
        try:
            df_display["date"] = pd.to_datetime(df_display["timestamp"], errors='coerce', unit='s').dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Dodaje kolumnę czasu gry i autora
    if "author" in df_display.columns:
        df_display["playtime_at_review_hours"] = df_display["author"].apply(
            lambda x: x.get("playtime_at_review_hours", 0) if isinstance(x, dict) else 0
        )
        df_display["author_display"] = df_display["author"].apply(lambda x: format_author_info(x) if isinstance(x, dict) else str(x))

    # Przygotowuje i wyświetla tabelę z konfiguracją kolumn
    st.dataframe(
        df_display.rename(columns={
            "tekst": "recenzja",
            "sentyment_label": "sentyment",
            "emocja_label": "emocja",
            "author_display": "autor",
            "playtime_at_review_hours": "czas_gry_h",
            "helpful": "pomocne",
            "sentyment_score": "score_sent",
            "emocja_score": "score_emo"
        })[["date", "sentyment", "score_sent", "emocja", "score_emo", "pomocne", "autor", "czas_gry_h", "recenzja"]],
        height=400,
        use_container_width=True,
        hide_index=True,
        column_config={
            "recenzja": st.column_config.TextColumn(width="medium"),
            "autor": st.column_config.TextColumn(width="small"),
            "sentyment": st.column_config.TextColumn(width="small"),
            "emocja": st.column_config.TextColumn(width="small"),
            "score_sent": st.column_config.NumberColumn(width="small", format="%.3f"),
            "score_emo": st.column_config.NumberColumn(width="small", format="%.3f"),
            "czas_gry_h": st.column_config.NumberColumn(width="small", format="%.1f h"),
            "pomocne": st.column_config.NumberColumn(width="small"),
            "date": st.column_config.TextColumn(width="small")
        }
    )
