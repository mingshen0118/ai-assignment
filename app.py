import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load data
@st.cache_data
def load_recipes():
    return pd.read_csv("https://drive.google.com/uc?id=1RQQk2fijSMUresMZnZSPN2OFBpx_4p2r", low_memory=False)

@st.cache_data
def load_reviews():
    return pd.read_csv("https://drive.google.com/uc?id=1xbHr4eftazl-005A56F-JufF4A9Rh5h9", low_memory=False)

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

st.title("🍽️ Recipe Recommender")

user_options = get_user_options()
selected_display = st.selectbox(
    "Select your profile:",
    options=[opt[0] for opt in user_options],
    index=0
)

recipe_name = st.text_input(
    "Enter a similar recipe to recommend:",
    placeholder="Type to search...",
    key="recipe_search"
)

if recipe_name:
    matches = [name for name in get_recipe_names()
              if recipe_name.lower() in name.lower()][:5]
    if matches:
        st.caption("Suggestions: " + ", ".join(matches))

author_id = next(opt[1] for opt in user_options if opt[0] == selected_display)
diet = st.selectbox("Dietary preference (optional):", ["None", "vegan", "keto", "gluten_free"])
diet_type = None if diet == "None" else diet

n = st.slider("How many recipes do you want recommended?", min_value=5, max_value=20, value=5)

sort_by_rating = st.checkbox("Sort recommendations by rating", value=True)

if st.button("Recommend"):
    with st.spinner("Finding best recipes for you..."):
        recommendations = hybrid_recommend(
            user_id=author_id,
            recipe_name=recipe_name if recipe_name != "None (show popular recipes)" else None,
            diet_type=diet_type,
            n=n,
            sort_by_rating=sort_by_rating
            )
        if recommendations is not None and not recommendations.empty:
            st.success("Here are your recommendations!")
            for idx, row in recommendations.iterrows():
              cols = st.columns([1, 4])
              with cols[0]:
                  st.metric("Rating", f"{row['avg_rating']:.1f}⭐")
              with cols[1]:
                  prep_time = int(row.get('PrepTime_min', 0))
                  cook_time = int(row.get('CookTime_min', 0))

                  time_parts = []
                  if prep_time > 0:
                      time_parts.append(f"⏳ {prep_time}m prep")
                  if cook_time > 0:
                      time_parts.append(f"🍳 {cook_time}m cook")

                  time_display = " • ".join(time_parts) if time_parts else "Time not specified"

                  st.markdown(f"""
                  <div style="font-size:2em; font-weight:bold;">{row['Name']}</div>
                  <div style="display:flex; justify-content:space-between; font-size:1em;">
                      <span>{time_display}</span>

                  </div>
                  <div style="display:flex; justify-content:space-between; font-size:0.95em; margin-top:4px;">
                      <span>💪 Protein: {row.get('protein', 0):.1f}g</span>
                      <span>🍞 Carbs: {row.get('carbs', 0):.1f}g</span>
                      <span>🧈 Fat: {row.get('fat', 0):.1f}g</span>
                      <span>🌾 Fiber: {row.get('fiber', 0):.1f}g</span>
                      <span>🔥 {int(row.get('calories', 0))} calories</span>
                  </div>
                  <div><br></div>
                  """, unsafe_allow_html=True)

                  raw_instructions = str(row.get('RecipeInstructions', ''))
                  instructions = re.findall(r'"(.*?)"', raw_instructions)

                  if instructions:
                      st.markdown("**Recipe Instructions:**")
                      instructions_text = "\n".join([f"{i}. {step}" for i, step in enumerate(instructions, 1)])
                      st.markdown(instructions_text)
                  else:
                      st.markdown("**Recipe Instructions:** Not available.")
        else:
            st.warning("No recommendations found.")
