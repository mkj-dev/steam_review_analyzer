from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import spacy

# Testowanie dostępności GPU
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# ------------- ASPEKTY GAMINGOWE – DWUJĘZYCZNE -------------
GAMING_ASPECTS = {
    "gameplay": {
        "en": ["gameplay", "mechanic", "mechanics", "control", "controls", "combat", "fighting", "ai", "difficulty", "level design", "mission", "quest", "character", "characters", "movement", "movement system"],
        "pl": ["gameplay", "mechanika", "mechaniki", "sterowanie", "walka", "ai", "trudność", "poziom", "misje", "quest", "questy", "postacie", "postać", "ruch", "system walki"]
    },
    "graphics": {
        "en": ["graphics", "visual", "visuals", "art", "art style", "texture", "textures", "model", "models", "animation", "animations", "fps", "frame rate", "resolution", "lighting", "shader", "shaders", "rendering"],
        "pl": ["grafika", "wygląd", "wizualny", "tekstury", "tekstura", "modele", "model", "animacje", "animacja", "fps", "rozdzielczość", "światło", "oświetlenie", "widok", "detale", "szader", "renderowanie"]
    },
    "performance": {
        "en": ["crash", "crashes", "bug", "bugs", "glitch", "glitches", "lag", "lags", "freeze", "freezes", "optimization", "optimized", "loading", "load time", "stutter", "stuttering", "memory", "performance", "fps drop", "frame drop"],
        "pl": ["crash", "crashuje", "bug", "błąd", "błędy", "lag", "zawiesza", "zawieszenie", "optymalizacja", "optymalizowana", "ładowanie", "freeze", "wydajność", "spadki", "spadek fps", "fps spada"]
    },
    "story": {
        "en": ["story", "plot", "narrative", "character", "characters", "dialogue", "dialogues", "quest", "quests", "lore", "writing", "script", "storyline", "plotline"],
        "pl": ["fabuła", "historia", "postacie", "postać", "dialogi", "dialog", "quest", "questy", "opowieść", "świat", "scenariusz", "lore", "narracja"]
    },
    "multiplayer": {
        "en": ["multiplayer", "multi-player", "server", "servers", "netcode", "connection", "connections", "pvp", "coop", "co-op", "matchmaking", "online", "network", "ping", "latency"],
        "pl": ["multiplayer", "serwer", "serwery", "sieć", "połączenie", "pvp", "kooperacja", "ko-op", "matchmaking", "online", "ping", "opóźnienie"]
    },
    "audio": {
        "en": ["sound", "sounds", "music", "voice", "voices", "voice acting", "audio", "effect", "effects", "ambient", "soundtrack", "ost", "sfx"],
        "pl": ["dźwięk", "dźwięki", "muzyka", "głos", "głosy", "audio", "efekty", "efekt", "muzyka", "ścieżka", "dubbing", "napisy", "ost"]
    },
    "ui": {
        "en": ["ui", "interface", "interfaces", "menu", "menus", "hud", "inventory", "font", "fonts", "usability", "ux", "user interface"],
        "pl": ["interfejs", "menu", "hud", "ekwipunek", "czcionka", "czcionki", "gui", "okno", "okna", "opcje", "ustawienia", "ui"]
    },
    "monetization": {
        "en": ["price", "pricing", "dlc", "dlcs", "microtransaction", "microtransactions", "pay", "paid", "expensive", "cheap", "value", "sale", "sales", "cost", "money", "purchase"],
        "pl": ["cena", "ceny", "dlc", "mikrotransakcje", "mikrotransakcja", "drogo", "drogi", "tanio", "tani", "wartość", "promocja", "promocje", "zawartość", "season pass", "płatne"]
    }
}

@st.cache_resource
def load_spacy_model(language: str = "english"):
    """
    Ładuje model spaCy.
    Zwraca: None, jeśli model nie jest zainstalowany.
    """
    try:
        model_name = "pl_core_news_sm" if language == "polish" else "en_core_web_sm"
        return spacy.load(model_name)
    except OSError:
        st.warning(f"Model spaCy {model_name} nie jest zainstalowany. Lematyzacja nie będzie działać.")
        return None

@st.cache_resource
def load_sentiment_model():
    """
    Ładuje model sentymentu.
    Zwraca: tokenizer, model, id2label, labels_map, device.
    """
    model_name = "clapAI/roberta-large-multilingual-sentiment"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        st.error(f"Nie udało się wczytać modelu sentymentu '{model_name}': {e}")
        raise

    # Jeżeli nie ma GPU to używamy CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Załadowano model sentymentu '{model_name}' na urządzenie {device}.")

    # Pobierz id2label z konfiguracji (gdy dostępne)
    # Dla clapAI/modernBERT-large-multilingual-sentiment jest to:
    # "id2label": { "0": "negative", "1": "neutral", "2": "positive" }
    id2label = {}
    try:
        id2label = {int(k): v for k, v in model.config.id2label.items()}
    except Exception:
        # fallback: jeżeli nie ma id2label, zbuduj domyślne (np. dla modelu 3-klasowego)
        id2label = {0: "neg", 1: "neu", 2: "pos"}

    # Mapowanie etykiet oryginalnych -> polskie
    labels_map = {}
    for idx, label in id2label.items():
        ll = label.lower()
        if "neg" in ll or "negative" in ll or "negaty" in ll:
            labels_map[label] = "negatywny"
        elif "pos" in ll or "positive" in ll or "pozy" in ll:
            labels_map[label] = "pozytywny"
        elif "neu" in ll or "neutral" in ll:
            labels_map[label] = "neutralny"
        else:
            # Zostawiamy oryginalną etykietę (do dalszego mapowania w analizie)
            labels_map[label] = label

    return tokenizer, model, id2label, labels_map, device

@st.cache_resource
def load_emotion_model(language: str = "english"):
    """
    Ładuje model emocji w zależności od języka.
    Zwraca: tokenizer, model, labels, device, label_mapping.
    """
    if language == "polish":
        model_name = "visegradmedia-emotion/Emotion_RoBERTa_pooled_V4"
        # Model zwraca etykiety po polsku
        labels = ["złość", "smutek", "strach", "radość", "neutralny", "zaskoczenie", "odraza"]
        label_mapping = None  # Etykiety już po polsku
    else:
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        label_mapping = {
            "anger": "złość",
            "disgust": "odraza",
            "fear": "strach",
            "joy": "radość",
            "neutral": "neutralny",
            "sadness": "smutek",
            "surprise": "zaskoczenie"
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        st.error(f"Nie udało się wczytać modelu emocji '{model_name}': {e}")
        raise

    # Jeżeli nie ma GPU to używamy CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Załadowano model emocji '{model_name}' na urządzenie {device}.")

    return tokenizer, model, labels, device, label_mapping

def _softmax(logits: np.ndarray) -> np.ndarray:
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def analyze_texts(reviews: List[Dict], tokenizer, model, labels: Any, device,
                  label_mapping: Optional[Dict] = None) -> Tuple[List[Dict], str]:
    """
    Analiza tekstów.
    Zwraca: wyniki, caly_tekst.
    """
    wyniki, caly_tekst = [], ""
    batch_size = 8

    # Rozpoznaj typ modelu
    is_sentiment_model = isinstance(labels, dict) and len(labels) in (2, 3, 5)

    # Mapowanie polaryzacji
    if is_sentiment_model:
        polarity_map = {"negative": "negatywny", "neutral": "neutralny", "positive": "pozytywny",
                        "negatywny": "negatywny", "pozytywny": "pozytywny", "neutralny": "neutralny"}
    else:  # emocje
        polarity_map = {"złość": "negatywny", "odraza": "negatywny", "strach": "negatywny", "smutek": "negatywny",
                        "radość": "pozytywny", "neutralny": "neutralny", "zaskoczenie": "neutralny",
                        "anger": "negatywny", "disgust": "negatywny", "fear": "negatywny", "joy": "pozytywny",
                        "sadness": "negatywny", "surprise": "neutralny"}

    for i in range(0, len(reviews), batch_size):
        batch_reviews = reviews[i:i + batch_size]
        batch_texts = [r.get("review", "") for r in batch_reviews]

        if not batch_texts:
            continue

        try:
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits.cpu().numpy()
                probs = _softmax(logits)
        except Exception as e:
            # Jeżeli wystąpi błąd to nie przerywamy całej aplikacji
            st.error(f"Błąd podczas inferencji modelu: {e}")
            # Fallback: dla każdego tekstu daj etykietę neutralną z niską pewnością
            for r_obj in batch_reviews:
                txt = r_obj.get("review", "")
                if not txt.strip():
                    continue
                wyniki.append({
                    "tekst": txt,
                    "etykieta_modelu": "neutralny",
                    "pewnosc": 0.5,
                    "sentyment_score": 0.5,
                    "timestamp": r_obj.get("timestamp"),
                    "polarity": "neutralny",
                    "helpful": r_obj.get("helpful"),
                    "purchase_type": r_obj.get("purchase_type"),
                    "voted_up": r_obj.get("voted_up"),
                    "author": r_obj.get("author"),
                    "weighted_vote_score": r_obj.get("weighted_vote_score"),
                    "comment_count": r_obj.get("comment_count"),
                    "early_access": r_obj.get("early_access"),
                    "received_for_free": r_obj.get("received_for_free"),
                    "language": r_obj.get("language", "english")
                })
            continue

        for r_obj, p in zip(batch_reviews, probs):
            txt = r_obj.get("review", "")
            if not txt.strip():
                continue

            idx = int(np.argmax(p))

            # Domyślny sentyment
            sentyment_score = 0.5

            # Pobieramy etykietę modelu
            if is_sentiment_model:
                # labels: dict id->label
                etykieta_raw = labels.get(idx, f"LABEL_{idx}")
                # Stosujemy mapowanie do polskich etykiet jeśli dostępne
                if label_mapping:
                    etykieta = label_mapping.get(etykieta_raw, etykieta_raw)
                else:
                    etykieta = etykieta_raw
            else:
                # labels: list
                etykieta_raw = labels[idx] if idx < len(labels) else str(idx)
                if label_mapping:
                    etykieta = label_mapping.get(etykieta_raw.lower(), etykieta_raw.lower())
                else:
                    etykieta = etykieta_raw

            # Obliczamy sentyment_score
            if is_sentiment_model:
                # Próbujemy odnaleźć indeks klasy pozytywnej w model.config.id2label
                positive_idx = None
                try:
                    id2label = model.config.id2label
                    for k, v in id2label.items():
                        if "pos" in v.lower() or "positive" in v.lower() or "pozy" in v.lower():
                            positive_idx = int(k)
                            break
                except Exception:
                    positive_idx = None

                if positive_idx is not None and positive_idx < len(p):
                    sentyment_score = float(p[positive_idx])
                else:
                    # Fallback: przypisz ważoną sumę z domyślnymi wartościami
                    if len(p) == 3:
                        sent_scores = [0.2, 0.5, 0.8]
                    else:
                        # rozłożenie liniowe 0..1
                        sent_scores = np.linspace(0, 1, num=len(p)).tolist()
                    sentyment_score = float(np.dot(p, sent_scores))
            else:
                sentyment_score = float(p[idx])

            pol = polarity_map.get(str(etykieta).lower(), "neutralny")
            caly_tekst += txt + " "

            wyniki.append({
                "tekst": txt,
                "etykieta_modelu": etykieta,
                "pewnosc": float(p[idx]),
                "sentyment_score": sentyment_score,
                "timestamp": r_obj.get("timestamp"),
                "polarity": pol,
                "helpful": r_obj.get("helpful"),
                "purchase_type": r_obj.get("purchase_type"),
                "voted_up": r_obj.get("voted_up"),
                "author": r_obj.get("author"),
                "weighted_vote_score": r_obj.get("weighted_vote_score"),
                "comment_count": r_obj.get("comment_count"),
                "early_access": r_obj.get("early_access"),
                "received_for_free": r_obj.get("received_for_free"),
                "language": r_obj.get("language", "english")
            })

    return wyniki, caly_tekst

@st.cache_data
def analyze_aspects(reviews: List[Dict], language: str = "english") -> Dict[str, Dict]:
    """
    Analiza aspektów z wykorzystaniem spaCy (jeśli dostępny).
    Zwraca: results (słownik aspekt -> metryki).
    """
    lang_code = "pl" if language == "polish" else "en"
    nlp = load_spacy_model(language)

    aspect_data = defaultdict(lambda: {"mentions": [], "s_scores": []})

    for rev in reviews:
        text = (rev.get("tekst") or rev.get("review") or "").lower()
        found = set()

        # Sprawdź każdy aspekt przez proste dopasowanie słów kluczowych
        for aspect, keywords_dict in GAMING_ASPECTS.items():
            keywords = keywords_dict.get(lang_code, [])
            if any(kw in text for kw in keywords):
                found.add(aspect)

        for asp in found:
            aspect_data[asp]["mentions"].append(rev)
            aspect_data[asp]["s_scores"].append(rev.get("sentyment_score", 0.5))

    results = {}
    for asp, data in aspect_data.items():
        scores = data["s_scores"]
        avg = float(np.mean(scores)) if scores else 0.5
        neg_count = len([s for s in scores if s < 0.4])
        priority = neg_count * len(scores) * (1 - avg) if avg > 0 else neg_count * len(scores)

        results[asp] = {
            "avg_sentiment": float(avg),
            "mention_count": len(data["mentions"]),
            "priority_score": float(priority),
            "sample_reviews": data["mentions"][:3],
            "all_reviews": data["mentions"]
        }

    return results

@st.cache_data
def summarize_reviews(merged: List[Dict], top_n: int = 5, lang: str = "english") -> Dict[str, List[Tuple[str, int]]]:
    """
    Podsumowanie fraz z lematyzacją (jeśli spaCy dostępny).
    """
    texts_pos, texts_neg = [], []

    for m in merged:
        pol = (m.get("sentyment_polar") or m.get("sentyment_label") or "").lower()
        txt = m.get("tekst") or m.get("review") or ""
        if "pozy" in pol or "positive" in pol:
            texts_pos.append(txt)
        elif "neg" in pol or "negative" in pol:
            texts_neg.append(txt)

    # Lemmatyzacja z wykorzystaniem cache'owanego spaCy
    def lemmatize_batch(texts):
        if not texts:
            return texts

        nlp = load_spacy_model(lang)
        if nlp:
            lemmed = []
            for t in texts:
                doc = nlp(t)
                lemmed.append(" ".join(tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_stop))
            return lemmed
        return texts

    stop_words = "english" if lang == "english" else ["i", "w", "na", "nie", "że", "to", "się", "z", "do", "o", "po", "dla", "jest", "jak", "ale", "co", "czy", "od", "za"]
    summaries = {"positive": [], "negative": []}

    try:
        if texts_pos:
            vec = CountVectorizer(ngram_range=(1, 2), stop_words=stop_words, max_features=1000)
            X = vec.fit_transform(lemmatize_batch(texts_pos))
            freqs = X.sum(axis=0).A1
            phrases = vec.get_feature_names_out()
            top = sorted(zip(phrases, freqs), key=lambda x: -x[1])[:top_n]
            summaries["positive"] = [(p, int(c)) for p, c in top]

        if texts_neg:
            vec = CountVectorizer(ngram_range=(1, 2), stop_words=stop_words, max_features=1000)
            X = vec.fit_transform(lemmatize_batch(texts_neg))
            freqs = X.sum(axis=0).A1
            phrases = vec.get_feature_names_out()
            top = sorted(zip(phrases, freqs), key=lambda x: -x[1])[:top_n]
            summaries["negative"] = [(p, int(c)) for p, c in top]
    except Exception as e:
        # W razie błędów w CountVectorizer nie przerywamy aplikacji
        st.warning(f"Nie udało się wygenerować podsumowania fraz: {e}")

    return summaries
