import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import Dict, Any

def build_plotly_sentiment_bar(df: pd.DataFrame):
    if "sentyment_label" in df.columns and not df["sentyment_label"].empty:
        counts = df["sentyment_label"].value_counts().reindex(["pozytywny", "neutralny", "negatywny"]).fillna(0)
        # Sprawdzamy, czy są jakieś dane do wyświetlenia
        if counts.sum() > 0:
            # Mapowanie kolorów dla każdego sentymentu
            color_map = {
                "pozytywny": "#2ecc71",  # Zielony
                "neutralny": "#3498db",  # Niebieski
                "negatywny": "#e74c3c"    # Czerwony
            }
            
            # Tworzymy wykres używając go.Figure dla lepszej kontroli
            fig = go.Figure(data=[
                go.Bar(
                    x=counts.index,
                    y=counts.values,
                    marker_color=[color_map.get(label, "#95a5a6") for label in counts.index],
                    text=counts.values,
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Rozkład sentymentu",
                xaxis_title="Sentyment",
                yaxis_title="Liczba",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(categoryorder="array", categoryarray=["pozytywny", "neutralny", "negatywny"])
            )
        else:
            # Pusty wykres jeśli wystąpi brak danych
            fig = go.Figure()
            fig.add_annotation(text="Brak danych do wyświetlenia", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Rozkład sentymentu", margin=dict(l=10, r=10, t=40, b=10))
    else:
        # Dodatkowy kod dla nieoczekiwanej struktury danych
        fig = go.Figure()
        fig.add_annotation(text="Brak kolumny 'sentyment_label' w danych", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Rozkład sentymentu", margin=dict(l=10, r=10, t=40, b=10))
    return fig

def build_plotly_emotion_bar(df: pd.DataFrame):
    if "emocja_label" in df.columns and not df["emocja_label"].empty:
        counts = df["emocja_label"].value_counts()
        # Tworzymy wykres z liczbami na słupkach
        fig = go.Figure(data=[
            go.Bar(
                x=counts.index,
                y=counts.values,
                text=counts.values,
                textposition='auto',
            )
        ])
        fig.update_layout(
            title="Rozkład emocji",
            xaxis_title="Emocja",
            yaxis_title="Liczba",
            margin=dict(l=10, r=10, t=40, b=10)
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych do wyświetlenia", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Rozkład emocji", margin=dict(l=10, r=10, t=40, b=10))
    return fig

def build_sentiment_trend(df: pd.DataFrame) -> go.Figure:
    """Trend sentymentu w czasie - wykres liniowy z punktami"""
    if "date" not in df.columns or "sentyment_score" not in df.columns:
        return go.Figure()

    tmp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(tmp["date"]):
        tmp["date"] = pd.to_datetime(tmp["date"], errors='coerce')

    tmp = tmp.dropna(subset=["date", "sentyment_score"])

    if tmp.empty:
        return go.Figure()

    # Agregujemy dane dzienne
    daily = tmp.groupby(tmp["date"].dt.date).agg(
        avg_sentiment=("sentyment_score", "mean"),
        count=("sentyment_score", "count")
    ).reset_index()

    if daily.empty:
        return go.Figure()

    # Dodajemy 7-dniową średnią ruchomą
    daily["ma7"] = daily["avg_sentiment"].rolling(window=7, min_periods=1).mean()

    fig = go.Figure()

    # Punkty dzienne
    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["avg_sentiment"],
        mode="markers",
        name="Dzienny sentyment",
        marker=dict(size=6, opacity=0.6)
    ))

    # Linia trendu (średnia 7-dniowa)
    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["ma7"],
        mode="lines",
        name="Trend (MA7)",
        line=dict(width=3)
    ))

    fig.update_layout(
        title="Trend sentymentu w czasie",
        xaxis_title="Data",
        yaxis_title="Średni sentyment",
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 1])  # Zakres 0-1 dla sentyment_score
    )

    return fig

def build_steam_vs_model_comparison(df: pd.DataFrame) -> go.Figure:
    """Wykres słupkowy: ocena Steam (voted_up) vs. sentyment modelu"""
    if "voted_up" not in df.columns or "sentyment_polar" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Brak wymaganych kolumn w danych", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Porównanie: Ocena Steam vs. Sentyment modelu", margin=dict(l=10, r=10, t=40, b=10))
        return fig

    # Usuwamy wiersze z brakującymi wartościami
    df_clean = df.dropna(subset=["voted_up", "sentyment_polar"])
    
    if df_clean.empty:
        fig = go.Figure()
        fig.add_annotation(text="Brak danych do wyświetlenia", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Porównanie: Ocena Steam vs. Sentyment modelu", margin=dict(l=10, r=10, t=40, b=10))
        return fig

    # Tworzymy macierz porównawczą
    comparison = df_clean.groupby(["voted_up", "sentyment_polar"]).size().unstack(fill_value=0)

    # Upewniamy się, że mamy wszystkie kategorie
    if comparison.empty:
        comparison = pd.DataFrame(index=[True, False], columns=["negatywny", "neutralny", "pozytywny"]).fillna(0)
    else:
        if True not in comparison.index:
            comparison.loc[True] = 0
        if False not in comparison.index:
            comparison.loc[False] = 0

        for col in ["negatywny", "neutralny", "pozytywny"]:
            if col not in comparison.columns:
                comparison[col] = 0

        # Sortujemy
        comparison = comparison.reindex([True, False], fill_value=0)
        comparison = comparison.reindex(columns=["negatywny", "neutralny", "pozytywny"], fill_value=0)

    # Tworzymy dane do wykresu
    categories = ["Steam Pozytywna", "Steam Negatywna"]
    neg_values = (comparison.loc[True, "negatywny"] if True in comparison.index else 0,
                  comparison.loc[False, "negatywny"] if False in comparison.index else 0)
    neu_values = (comparison.loc[True, "neutralny"] if True in comparison.index else 0,
                  comparison.loc[False, "neutralny"] if False in comparison.index else 0)
    pos_values = (comparison.loc[True, "pozytywny"] if True in comparison.index else 0,
                  comparison.loc[False, "pozytywny"] if False in comparison.index else 0)

    fig = go.Figure()
    
    # Negatywny - czerwony
    fig.add_trace(go.Bar(
        name="Negatywny",
        x=categories,
        y=neg_values,
        marker_color="#e74c3c",
        text=neg_values,
        textposition='auto',
        hovertemplate="<b>Model Negatywny</b><br>%{x}<br>Liczba recenzji: %{y}<extra></extra>"
    ))
    
    # Neutralny - niebieski
    fig.add_trace(go.Bar(
        name="Neutralny",
        x=categories,
        y=neu_values,
        marker_color="#3498db",
        text=neu_values,
        textposition='auto',
        hovertemplate="<b>Model Neutralny</b><br>%{x}<br>Liczba recenzji: %{y}<extra></extra>"
    ))
    
    # Pozytywny - zielony
    fig.add_trace(go.Bar(
        name="Pozytywny",
        x=categories,
        y=pos_values,
        marker_color="#2ecc71",
        text=pos_values,
        textposition='auto',
        hovertemplate="<b>Model Pozytywny</b><br>%{x}<br>Liczba recenzji: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title="Porównanie: Ocena Steam vs. Sentyment modelu",
        xaxis_title="Ocena Steam",
        yaxis_title="Liczba recenzji",
        barmode="stack",
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig

def build_wordcloud_figure(text: str):
    wc = WordCloud(width=900, height=450, background_color='white', collocations=False).generate(text or "")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    return fig

def build_nps_gauge(nps_score: float) -> go.Figure:
    """Net Promoter Score – miernik lojalności graczy -100...+100."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=nps_score,
        title={'text': "Metryka NPS", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkgreen" if nps_score > 0 else "darkred"},
            'steps': [
                {'range': [-100, 0], 'color': "lightcoral"},
                {'range': [0, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        },
        number={'font': {'size': 40}}
    ))
    fig.update_layout(
        margin=dict(l=50, r=50, t=60, b=50),
        height=300
    )
    return fig

def build_aspect_bar(aspect_df: pd.DataFrame) -> go.Figure:
    """Wykres słupkowy: aspekt -> avg_sentiment."""
    # Mapowanie kolorów na podstawie wartości sentymentu (0-1)
    colors = []
    for val in aspect_df["avg_sentiment"]:
        if val >= 0.7:
            colors.append("#2ecc71")  # Zielony dla pozytywnych
        elif val >= 0.5:
            colors.append("#3498db")  # Niebieski dla neutralnych
        else:
            colors.append("#e74c3c")  # Czerwony dla negatywnych
    
    fig = go.Figure(data=[
        go.Bar(
            x=aspect_df.index,
            y=aspect_df["avg_sentiment"],
            text=[f"{v:.2f}" for v in aspect_df["avg_sentiment"]],
            textposition='auto',
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Śr. sentyment: %{y:.2f}<br>Wzmianek: %{customdata}<extra></extra>",
            customdata=aspect_df.get("mention_count", [0] * len(aspect_df))
        )
    ])
    fig.update_layout(
        title="Średni sentyment per aspekt (gaming)",
        xaxis_title="Aspekt",
        yaxis_title="Śr. sentyment",
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 1])
    )
    return fig

def build_aspect_bar_from_dict(aspects: Dict[str, Any]) -> go.Figure:
    """
    Przyjmuje słownik aspects zwrócony przez analyze_aspects i tworzy DataFrame
    do wykresu.
    """
    if not aspects:
        return go.Figure()

    df = pd.DataFrame.from_dict({k: {"avg_sentiment": v.get("avg_sentiment", 0.5),
                                     "mention_count": v.get("mention_count", 0)} for k, v in aspects.items()},
                                orient='index')
    if df.empty:
        return go.Figure()
    # Sortuj według liczby wzmianek (malejąco) dla lepszej czytelności
    df = df.sort_values("mention_count", ascending=False)
    return build_aspect_bar(df)
