import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load data
@st.cache_data
def load_recipes():
    return pd.read_csv("preprocessed_recipes.csv")

@st.cache_data
def load_reviews():
    try:
        return pd.read_csv("reviews.csv")
    except Exception as e:
        st.error(f"Failed to load reviews: {e}")
        return pd.DataFrame()

recipes = load_recipes()
reviews = load_reviews()


