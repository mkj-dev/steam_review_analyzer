import requests
from typing import List, Dict
import streamlit as st
import time
from urllib.parse import quote_plus

@st.cache_data
def search_game_by_title(title: str) -> List[Dict[str, str]]:
    if not title:
        return []
    url_title = quote_plus(title)
    url = f"https://steamcommunity.com/actions/SearchApps/{url_title}"
    max_attempts, backoff = 3, 1.0

    session = requests.Session()
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            results = response.json()
            return [{"name": r.get("name", ""), "appid": r.get("appid")} for r in results]
        except requests.RequestException as e:
            if attempt == max_attempts:
                st.error(f"Błąd podczas wyszukiwania gry: {e}")
                return []
            time.sleep(backoff)
            backoff *= 2.0
        except ValueError as e:
            # Problem z parsowaniem JSON
            st.error(f"Niepoprawna odpowiedź serwera podczas wyszukiwania gry: {e}")
            return []
    return []

def fetch_reviews_page(app_id: str, cursor: str, language: str, session: requests.Session = None) -> Dict:
    """
    Pobiera jedną stronę recenzji – zwraca surową odpowiedź JSON.
    session: opcjonalna sesja do ponownego użycia połączeń HTTP.
    Funkcja nie jest cache'owana, ponieważ jest wywoływana z cache'owanej funkcji fetch_reviews.
    """
    # Poprawione łączenie stringa URL (bez złamania f-string)
    base_url = (
        f"https://store.steampowered.com/appreviews/{app_id}"
        f"?json=1&language={language}&filter=recent&purchase_type=all"
    )
    url = f"{base_url}&num_per_page=100&cursor={quote_plus(cursor)}"
    max_attempts, backoff = 3, 1.0

    # Użyj istniejącej sesji lub utwórz nową
    if session is None:
        session = requests.Session()
        close_session = True
    else:
        close_session = False

    try:
        for attempt in range(1, max_attempts + 1):
            try:
                response = session.get(url, timeout=15)
                response.raise_for_status()
                # Jeżeli serwer zwraca nagłówek Retry-After, odczekaj
                retry_after = int(response.headers.get("Retry-After", 0) or 0)
                if retry_after:
                    time.sleep(retry_after)
                try:
                    return response.json()
                except ValueError:
                    st.error("Otrzymano niepoprawny format JSON z API Steam.")
                    return {}
            except requests.RequestException as e:
                if attempt == max_attempts:
                    st.error(f"Błąd podczas pobierania recenzji: {e}")
                    return {}
                time.sleep(backoff)
                backoff *= 2.0
        return {}
    finally:
        if close_session:
            session.close()

@st.cache_data
def fetch_reviews(app_id: str, number_of_reviews: int, language: str) -> List[Dict]:
    """
    Pobiera recenzje dla app_id, zwracając listę słowników z oczyszczonymi metadanymi.
    Zoptymalizowane z ponownym użyciem sesji HTTP i zmniejszonymi opóźnieniami.
    """
    if not app_id:
        return []

    all_reviews, cursor, fetched = [], "*", 0
    target = min(number_of_reviews, 1000)
    
    # Reuse session across all requests for better performance
    session = requests.Session()
    
    try:
        while fetched < target:
            data = fetch_reviews_page(app_id, cursor, language, session)
            reviews = data.get("reviews", []) if isinstance(data, dict) else []

            if not reviews:
                break

            for r in reviews:
                ts = r.get("timestamp_created")
                if not ts:
                    # pomiń recenzje bez timestampu
                    continue

                # Parsowanie typu zakupu
                steam_purchase = r.get("steam_purchase", True)
                purchase_type = "steam" if steam_purchase else "key"

                # Pobierz pełne dane autora
                author_dict = r.get("author", {}) or {}
                author_info = {}
                if isinstance(author_dict, dict):
                    # Konwertuj czas gry z minut na godziny (1 decimal)
                    try:
                        playtime_at_review_hours = round(int(author_dict.get("playtime_at_review", 0)) / 60, 1)
                        playtime_forever_hours = round(int(author_dict.get("playtime_forever", 0)) / 60, 1)
                    except Exception:
                        playtime_at_review_hours = 0.0
                        playtime_forever_hours = 0.0

                    author_info = {
                        "steamid": author_dict.get("steamid", "N/A"),
                        "num_games_owned": int(author_dict.get("num_games_owned", 0) or 0),
                        "num_reviews": int(author_dict.get("num_reviews", 0) or 0),
                        "playtime_at_review_hours": playtime_at_review_hours,
                        "playtime_forever_hours": playtime_forever_hours
                    }

                # Dodaj recenzję z pełnymi metadanymi
                try:
                    weighted = float(r.get("weighted_vote_score", 0) or 0)
                except Exception:
                    weighted = 0.0

                all_reviews.append({
                    "review": r.get("review", "") or "",
                    "timestamp": int(ts),
                    "helpful": int(r.get("votes_up", 0) or 0),
                    "purchase_type": purchase_type,
                    "voted_up": bool(r.get("voted_up", False)),
                    "author": author_info,
                    "weighted_vote_score": weighted,
                    "comment_count": int(r.get("comment_count", 0) or 0),
                    "early_access": bool(r.get("written_during_early_access", False)),
                    "received_for_free": bool(r.get("received_for_free", False))
                })

                fetched += 1
                if fetched >= target:
                    break

            cursor = data.get("cursor") if isinstance(data, dict) else None
            if not cursor or len(reviews) < 100:
                break

            # Zmniejszone opóźnienie - Steam API zwykle toleruje 0.05-0.1s
            # Dla większych batchy używamy mniejszego opóźnienia
            time.sleep(0.05)
    finally:
        session.close()

    if not all_reviews:
        st.warning("Nie udało się pobrać żadnych recenzji. Możliwe ograniczenia API Steam.")
        return []

    st.caption(f"Pobrano {len(all_reviews)} recenzji (najświeższe dostępne).")
    return all_reviews
