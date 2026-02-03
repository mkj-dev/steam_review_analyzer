# steam_review_analyzer
Aplikacja Streamlit do analizy sentymentu i emocji w recenzjach gier pobranych z platformy Steam. Narzędzie automatycznie pobiera recenzje użytkowników, wykonuje ich analizę i prezentuje wyniki w formie interaktywnych wizualizacji.
W aplikacji wykorzystano trzy modele:
- **modernBERT-large-multilingual-sentiment** do analizy sentymentu w języku polskim i angielskim (https://huggingface.co/clapAI/roberta-large-multilingual-sentiment)
- **Emotion_RoBERTa_pooled_V4** do analizy emocji w języku polskim (https://huggingface.co/visegradmedia-emotion/Emotion_RoBERTa_pooled_V4)
- **emotion-english-distilroberta-base** do analizy emocji w języku angielskim (https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

## Funkcje

- **Dwujęzyczna analiza**: Wsparcie dla recenzji w języku polskim i angielskim
- **Analiza sentymentu**: Klasyfikacja na pozytywny, neutralny i negatywny
- **Analiza emocji**: Rozpoznawanie emocji: radość, złość, smutek, strach, zaskoczenie, odraza, neutralny
- **Analiza aspektów gamingowych**: Automatyczne wykrywanie i ocena aspektów (gameplay, grafika, wydajność, fabuła, multiplayer, audio, UI, monetyzacja)
- **Net Promoter Score (NPS)**: Metryka lojalności graczy w skali -100 do +100
- **Filtrowanie spamu**: Automatyczne odfiltrowywanie recenzji niskiej jakości, wulgaryzmów i losowych znaków
- **Interaktywne wizualizacje**: Wykresy Plotly, chmura słów, trendy czasowe, porównanie z ocenami Steam
- **Eksport danych**: Możliwość pobrania wyników w formacie CSV i JSON

## Instalacja pakietów Python - wymagany Python 3.13
```pip install -r requirements.txt```

## Instalacja modeli językowych spaCy
```python -m spacy download pl_core_news_sm```<br>
```python -m spacy download en_core_web_sm```<br>
**Bez zainstalowania modeli spaCy, aplikacja będzie działać, ale lematyzacja i analiza aspektów będą ograniczone.**

## Pierwsze uruchomienie
```streamlit run app.py lub streamlit -m run app.py```

## Uwaga
Przy pierwszym uruchomieniu aplikacja automatycznie pobierze modele z Hugging Face:
- Model sentymentu: clapAI/roberta-large-multilingual-sentiment
- Model emocji (polski): visegradmedia-emotion/Emotion_RoBERTa_pooled_V4
- Model emocji (angielski): j-hartmann/emotion-english-distilroberta-base<br>
**Pobranie modeli może zająć kilka minut i wymaga stabilnego połączenia internetowego.**
