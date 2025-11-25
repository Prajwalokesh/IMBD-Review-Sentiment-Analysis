"""Streamlit app for IMDB review sentiment analysis using a pre-trained Simple RNN model."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Simple_rnn_imbd.h5"
MAXLEN = 500
THRESHOLD = 0.5


class SimpleRNNDowngrade(SimpleRNN):
    """Compatibility layer to ignore the deprecated `time_major` arg when loading."""

    def __init__(self, *args, time_major=False, **kwargs):  # type: ignore[override]
        super().__init__(*args, **kwargs)


@st.cache_resource(show_spinner=False)
def load_sentiment_model(model_path: Path):
    """Load and cache the trained sentiment model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Ensure 'Simple_rnn_imbd.h5' is available."
        )
    return load_model(model_path, custom_objects={"SimpleRNN": SimpleRNNDowngrade})


@st.cache_data(show_spinner=False)
def load_imdb_indices() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Fetch and cache the IMDB word index mappings."""
    word_index = imdb.get_word_index()
    reverse_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_index


def preprocess_text(text: str, word_index: Dict[str, int], maxlen: int = MAXLEN) -> np.ndarray:
    """Convert raw text into a padded sequence compatible with the model."""
    tokens = re.findall(r"[\w']+", text.lower())
    encoded = [word_index.get(token, 2) + 3 for token in tokens]

    if not encoded:
        return np.zeros((1, maxlen), dtype="int32")

    return sequence.pad_sequences([encoded], maxlen=maxlen)


def main():
    st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon=None, layout="centered")

    st.title("IMDB Review Sentiment Analyzer")
    st.write(
        "Enter a movie review to predict the sentiment using the pre-trained Simple RNN model. "
        "The model outputs a probability score and a Positive/Negative label."
    )

    word_index, _ = load_imdb_indices()

    default_review = (
        "This movie was fantastic! The performances were powerful and the pacing kept me engaged."
    )
    user_review = st.text_area("Movie review", value=default_review, height=200)

    analyze = st.button("Analyze Sentiment", type="primary")

    if analyze:
        if not user_review.strip():
            st.warning("Please enter a review before running the analysis.")
            st.stop()

        with st.spinner("Running inference..."):
            try:
                model = load_sentiment_model(MODEL_PATH)
            except FileNotFoundError as err:
                st.error(str(err))
                st.stop()

            processed = preprocess_text(user_review, word_index)
            prediction = model.predict(processed, verbose=0)
            score = float(prediction[0][0])
            sentiment = "Positive" if score > THRESHOLD else "Negative"

        st.success(f"Sentiment: {sentiment}")
        st.metric(label="Confidence", value=f"{score:.2%}")

        st.caption(
            "Confidence represents the model's probability for the positive class. "
            "Values above 50% are labeled Positive."
        )

    with st.expander("Need inspiration?"):
        st.write(
            "Try phrases like: 'The plot was slow and predictable' or "
            "'I loved the characters and the soundtrack.'"
        )


if __name__ == "__main__":
    main()
