import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PASSWORD PROTECTION ---
PASSWORD = "geonly123"
pwd = st.text_input("Enter password", type="password")
if pwd != PASSWORD:
    st.warning("Access denied.")
    st.stop()

# --- FILE UPLOADER ---
st.title("GE SKU Matching Tool with Smart Feature Weighting")
uploaded_file = st.file_uploader("Upload your SKU Excel file", type=["xlsx", "xls"])
if not uploaded_file:
    st.stop()

# --- LOAD AND TRANSPOSE FILE ---
df_raw = pd.read_excel(uploaded_file, header=None)
df = df_raw.T
df.columns = df.iloc[0]
df = df[1:]
df.reset_index(drop=True, inplace=True)
df.fillna('', inplace=True)

# Ensure SKU column
if 'SKU' not in df.columns:
    df.rename(columns={df.columns[0]: 'SKU'}, inplace=True)
df['SKU'] = df['SKU'].astype(str)

# --- AUTO-DETECT PRODUCT CATEGORY ---
product_category = None
for col in df.columns:
    if df[col].astype(str).str.contains("cooktop|range|fridge|refrigerator|dishwasher|microwave", case=False).any():
        product_category = df[col].iloc[0].strip().lower()
        break

# --- CATEGORY-SPECIFIC FEATURE OPTIONS ---
feature_options = {
    "refrigerator": ["Width", "Configuration", "Depth Type", "Panel Ready", "Dispenser Type", "Color", "Capacity"],
    "cooktop": ["Width", "Fuel Type", "Number of Burners", "Control Type", "Color", "Brand"],
    "range": ["Width", "Fuel Type", "Configuration", "Oven Capacity", "Color", "Brand"],
    "dishwasher": ["Width", "Sound Rating", "Panel Ready", "Tub Material", "Rack System", "Control Location"],
    "microwave": ["Type", "Installation", "Wattage", "Color", "Control Type"]
}

# Pick best-matching category
category_name = "general"
for key in feature_options:
    if key in (product_category or ""):
        category_name = key
        break

available_features = [f for f in feature_options.get(category_name, []) if f in df.columns]

# --- USER SELECTS MOST IMPORTANT FEATURES ---
st.subheader("Feature Matching Preferences")
st.markdown(f"**Detected category:** `{category_name.title()}`")
important_features = st.multiselect(
    "Which features are most important to match?",
    options=available_features,
    default=available_features[:2]
)

# --- COMBINE TEXT FOR TF-IDF ---
df['combined_specs'] = ""
for col in df.columns:
    weight = 3 if col in important_features else 1
    df['combined_specs'] += ((df[col].astype(str) + " ") * weight)

# --- TF-IDF MODEL ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_specs'])

# --- MATCHING LOGIC ---
def match_skus(input_sku, brand_filter="ge", top_n=5):
    input_row = df[df['SKU'] == input_sku]
    if input_row.empty:
        return f"SKU {input_sku} not found."

    input_index = input_row.index[0]
    input_vector = tfidf_matrix[input_index]
    similarities = cosine_similarity(input_vector, tfidf_matrix)[0]

    brand_col = 'Brand' if 'Brand' in df.columns else 'spec_14'
    config_col = 'Configuration' if 'Configuration' in df.columns else 'spec_7'
    status_col = 'Model Status' if 'Model Status' in df.columns else 'spec_9'
    description_col = 'Description' if 'Description' in df.columns else None

    input_config = input_row.iloc[0][config_col]
    df_copy = df.copy()
    df_copy['similarity'] = similarities.astype(float)

    if brand_filter == "ge":
        filtered = df_copy[
            (df_copy[brand_col].str.lower() == "ge") &
            (df_copy[config_col].str.lower() == input_config.lower()) &
            (df_copy['SKU'] != input_sku)
        ]
    else:
        filtered = df_copy[
            (df_copy[brand_col].str.lower() != "ge") &
            (df_copy[config_col].str.lower() == input_config.lower()) &
            (df_copy['SKU'] != input_sku)
        ]

    filtered = filtered.sort_values(by='similarity', ascending=False)

    columns_to_return = ['SKU', brand_col]
    if description_col: columns_to_return.append(description_col)
    columns_to_return += [config_col, status_col, 'similarity']

    rename_dict = {
        brand_col: 'Brand',
        config_col: 'Configuration',
        status_col: 'Model Status',
        'similarity': 'Similarity Score',
        description_col: 'Description' if description_col else ''
    }

    return filtered[columns_to_return].rename(columns=rename_dict).head(top_n)

# --- UI Inputs ---
input_sku = st.text_input("Enter a competitor SKU:")
match_type = st.selectbox("What kind of match do you want?", ["GE only", "Competitor (non-GE)"])

# --- Trigger Match ---
if input_sku:
    result_df = match_skus(input_sku, brand_filter="ge" if match_type == "GE only" else "non-ge")

    if isinstance(result_df, pd.DataFrame):
        result_df = result_df.reset_index(drop=True)
        for col in result_df.columns:
            result_df[col] = result_df[col].apply(lambda x: str(x) if pd.notnull(x) else '')
        st.table(result_df)
    else:
        st.warning(result_df)
