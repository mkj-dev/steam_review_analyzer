import re
from typing import List, Dict

# Lista wulgaryzmów (angielskie i polskie) - podstawowe
CURSE_WORDS = {
    "en": ["fuck", "shit", "cunt", "bitch", "asshole", "damn", "hell", "piss", "crap", "dick", "cock", "pussy"],
    "pl": ["kurwa", "chuj", "dupa", "pierdol", "jebany", "pierdolony", "sukinsyn", "skurwysyn", "dupka"]
}

def is_spam_review(review: Dict, language: str = "english") -> bool:
    """
    Sprawdza czy recenzja jest spamem lub niskiej jakości.
    
    Kryteria:
    - Zbyt krótka (< 10 znaków)
    - Tylko wulgaryzmy
    - Tylko losowe znaki/cyfry
    - Brak sensownych słów
    """
    text = (review.get("tekst") or review.get("review") or "").strip()
    
    # Zbyt krótka
    if len(text) < 10:
        return True
    
    # Sprawdź czy zawiera tylko wulgaryzmy
    lang_code = "pl" if language == "polish" else "en"
    curse_list = CURSE_WORDS.get(lang_code, [])
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    if not words:
        return True
    
    # Jeśli wszystkie słowa to wulgaryzmy lub bardzo krótkie (1-2 znaki)
    meaningful_words = [w for w in words if len(w) > 2 and w not in curse_list]
    if len(meaningful_words) == 0 and len(words) > 0:
        return True
    
    # Sprawdź czy to tylko losowe znaki/cyfry (więcej niż 50% to cyfry lub znaki specjalne)
    alphanumeric = re.findall(r'[a-zA-Z]', text)
    if len(alphanumeric) < len(text) * 0.3:  # Mniej niż 30% to litery
        return True
    
    # Sprawdź czy zawiera tylko powtarzające się znaki (np. "aaaa", "1111")
    if len(set(text.replace(" ", ""))) <= 2 and len(text.replace(" ", "")) > 3:
        return True
    
    return False

def filter_spam_reviews(reviews: List[Dict], language: str = "english") -> tuple[List[Dict], List[Dict]]:
    """
    Filtruje recenzje na normalne i spam.
    Zwraca: (normalne_recenzje, spam_recenzje)
    """
    normal_reviews = []
    spam_reviews = []
    
    for review in reviews:
        if is_spam_review(review, language):
            spam_reviews.append(review)
        else:
            normal_reviews.append(review)
    
    return normal_reviews, spam_reviews

def filter_curse_words_from_text(text: str, language: str = "english") -> str:
    """
    Usuwa wulgaryzmy z tekstu dla word cloud.
    """
    if not text:
        return text
    
    lang_code = "pl" if language == "polish" else "en"
    curse_list = CURSE_WORDS.get(lang_code, [])
    
    words = text.split()
    filtered_words = []
    
    for word in words:
        word_lower = word.lower().strip('.,!?;:()[]{}"\'')
        if word_lower not in curse_list:
            filtered_words.append(word)
    
    return " ".join(filtered_words)

