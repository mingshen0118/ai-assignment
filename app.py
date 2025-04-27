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

#TFIDF
@st.cache_resource
def precompute_tfidf(recipes):
    tfidf_model = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf_model.fit_transform(recipes['text_features'].fillna(''))
    return tfidf_model, tfidf_matrix

tfidf_model, tfidf_matrix = precompute_tfidf(recipes)

@st.cache_data
def hybrid_recommend(user_id=None, recipe_name=None, diet_type=None, n=5, sort_by_rating=True):
    global tfidf_model, tfidf_matrix
    try:
        # ===== CONTENT-BASED =====
        content_results = pd.DataFrame()
        if recipe_name:

            if tfidf_model is None or tfidf_matrix is None:
                tfidf_model, tfidf_matrix = precompute_tfidf(recipes)

            escaped_name = re.escape(recipe_name)

            matches = recipes[recipes['Name'].str.contains(escaped_name, case=False, na=False)]
            if not matches.empty:
                input_idx = matches.index[0]
                cosine_sim = cosine_similarity(tfidf_matrix[input_idx], tfidf_matrix)
                sim_scores = sorted(enumerate(cosine_sim[0]), key=lambda x: x[1], reverse=True)
                content_indices = [i[0] for i in sim_scores]
                content_results = recipes.iloc[content_indices].copy()

        # ===== COLLABORATIVE FILTERING =====
        collab_results = pd.DataFrame()
        if user_id is not None:
            user_reviews = reviews[reviews['AuthorId'] == user_id]

            if not user_reviews.empty:
                top_rated = user_reviews.nlargest(3, 'Rating')['RecipeId'].tolist()

                similar_users = reviews[
                    (reviews['RecipeId'].isin(top_rated)) &
                    (reviews['Rating'] >= 4) &
                    (reviews['AuthorId'] != user_id)
                ]['AuthorId'].unique()

                if len(similar_users) > 0:
                    collab_recs = reviews[
                        (reviews['AuthorId'].isin(similar_users)) &
                        (reviews['Rating'] >= 4)
                    ].groupby('RecipeId')['Rating'].agg(['mean', 'count'])

                    collab_recs = collab_recs[collab_recs['count'] >= 2].nlargest(2*n, 'mean')
                    collab_results = recipes[recipes['RecipeId'].isin(collab_recs.index)]

                    if diet_type and f'is_{diet_type}' in collab_results.columns:
                        collab_results = collab_results.sort_values(
                            by=[f'is_{diet_type}', 'avg_rating'],
                            ascending=[False, False]
                        )

        # ===== COMBINE RESULTS =====
        results = pd.concat([content_results, collab_results]).drop_duplicates('RecipeId')

        if diet_type and f'is_{diet_type}' in recipes.columns:
            results = results[results[f'is_{diet_type}'] == True]

        if results.empty:
            results = recipes.copy()
            if diet_type:
                results = results[results[f'is_{diet_type}'] == True]

        # User chooses to sort by rating
        if sort_by_rating:
          results = results.sort_values('avg_rating', ascending=False)
        results = results[results['calories'].between(50, 2000)]

        output_cols = ['Name', 'calories',  'protein', 'carbs', 'fat', 'fiber', 'avg_rating', 'CookTime_min', 'PrepTime_min', 'RecipeInstructions']
        if diet_type:
            output_cols.append(f'is_{diet_type}')

        return results[output_cols].head(n)

    except Exception as e:
        st.write(f"**Error:** `{e}` (Type: `{type(e)}`)")
        return recipes[['Name', 'calories', 'avg_rating', 'CookTime_min', 'PrepTime_min', 'RecipeInstructions']].nlargest(n, 'avg_rating')

@st.cache_data
def get_recipe_names():
    names = recipes['Name'].dropna().unique().tolist()
    names.insert(0, "None (show popular recipes)")
    return names

@st.cache_data
def get_user_options():
    user_stats = reviews.groupby(['AuthorId', 'AuthorName']).size().reset_index(name='ReviewCount')

    user_stats['display'] = user_stats.apply(
        lambda row: f"{row['AuthorName']} ({row['ReviewCount']} reviews)",
        axis=1
    )

    options = user_stats.sort_values('AuthorName')[['display', 'AuthorId']].values.tolist()
    options.insert(0, ["New User (no profile)", None])

    return options


