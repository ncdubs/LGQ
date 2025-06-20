import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- PASSWORD PROTECTION ---
PASSWORD = "geonly123"  # Change this later!
pwd = st.text_input("Enter password", type="password")
if pwd != PASSWORD:
    st.warning("Access denied.")
    st.stop()

# --- FILE UPLOADER ---
st.title("GE SKU Matching Tool")
uploaded_file = st.file_uploader("Upload your SKU Excel file", type=["xlsx", "xls"])
if not uploaded_file:
    st.stop()

# --- LOAD FILE AND TRANSPOSE ---
df_raw = pd.read_excel(uploaded_file, header=None)
df = df_raw.T
df.columns = df.iloc[0]
df = df[1:]
df.reset_index(drop=True, inplace=True)

# --- DETECT & INSERT DESCRIPTION ROW IF NEEDED ---
keywords = ["cu ft", "side by side", "french door", '"', "in.", "top freezer", "bottom freezer", "refrigerator"]
searchable = df.applymap(lambda x: str(x).lower())
row_scores = []
for i, row in searchable.iterrows():
    score = sum(any(kw in cell for kw in keywords) for cell in row)
    if score > 2:
        row_scores.append((i, score))

if row_scores:
    best_row_index = sorted(row_scores, key=lambda x: x[1], reverse=True)[0][0]
    description_row = df.iloc[best_row_index]
    df_raw_rebuilt = pd.concat([
        df_raw.iloc[:2],
        pd.DataFrame([["Description"] + description_row.tolist()]),
        df_raw.iloc[2:]
    ]).reset_index(drop=True)
    df = df_raw_rebuilt.T
    df.columns = df.iloc[0]
    df = df[1:]
    df.reset_index(drop=True, inplace=True)

# --- BASIC CLEANING ---
if 'SKU' not in df.columns:
    df.rename(columns={df.columns[0]: 'SKU'}, inplace=True)
df.fillna('', inplace=True)
df['SKU'] = df['SKU'].astype(str)

# --- DETECT PRODUCT TYPE ---
product_keywords = {
    "Refrigerators": ["refrigerator", "fridge", "cu ft", "freezer", "side by side"]
}

detected_product = "General"
for keyword in product_keywords["Refrigerators"]:
    if df.apply(lambda row: row.astype(str).str.lower().str.contains(keyword).any(), axis=1).any():
        detected_product = "Refrigerators"
        break

# --- PRODUCT-SPECIFIC FEATURES ---
feature_options = {
    "Refrigerators": [
        "Refrigerator Depth",
        "Width (Approx.)",
        "Depth",
        "Standard Color",
        "Ice and Water Options",
        "Capacity (cu ft)",
        "Energy Rating",
        "Wifi Connected",
        "ADA Compliant"
    ]
}

available_features = [f for f in feature_options.get(detected_product, []) if f in df.columns]

# --- FEATURE IMPORTANCE SELECTION ---
st.subheader("Feature Matching Preferences")
st.markdown(f"**Detected Product Type:** `{detected_product}`")
important_features = st.multiselect(
    "Which features are most important to match?",
    options=available_features,
    default=available_features[:3]
)

# --- COMBINE SPEC TEXT BASED ON WEIGHTS ---
df['combined_specs'] = ""
for col in df.columns:
    weight = 3 if col in important_features else 1
    df['combined_specs'] += ((df[col].astype(str) + " ") * weight)

# --- TF-IDF MODEL ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_specs'])

# --- MATCHING FUNCTION ---
def find_matches(input_sku, brand_filter='ge', top_n=5):
    input_row = df[df['SKU'] == input_sku]
    if input_row.empty:
        return f"SKU {input_sku} not found."

    input_index = input_row.index[0]
    similarities = cosine_similarity(tfidf_matrix[input_index], tfidf_matrix)[0]

    brand_col = 'Brand' if 'Brand' in df.columns else 'spec_14'
    config_col = 'Configuration' if 'Configuration' in df.columns else 'spec_7'
    status_col = 'Model Status' if 'Model Status' in df.columns else 'spec_9'
    description_col = 'Description' if 'Description' in df.columns else None

    input_config = input_row.iloc[0][config_col]
    df_copy = df.copy()
    df_copy['similarity'] = similarities.astype(str)

    if brand_filter == "ge":
        filtered = df_copy[
            (df_copy[brand_col].str.lower() == 'ge') &
            (df_copy[config_col].str.lower() == input_config.lower()) &
            (df_copy['SKU'] != input_sku)
        ]
    else:
        filtered = df_copy[
            (df_copy[brand_col].str.lower() != 'ge') &
            (df_copy[config_col].str.lower() == input_config.lower()) &
            (df_copy['SKU'] != input_sku)
        ]

    filtered = filtered.sort_values(by='similarity', ascending=False)

    columns_to_return = ['SKU', brand_col]
    if description_col: columns_to_return.append(description_col)
    columns_to_return += [config_col, status_col]

    rename_dict = {
        brand_col: 'Brand',
        config_col: 'Configuration',
        status_col: 'Model Status'
    }
    if description_col:
        rename_dict[description_col] = 'Description'

    return filtered[columns_to_return].rename(columns=rename_dict).head(top_n)

# --- UI INPUTS ---
input_sku = st.text_input("Enter a competitor SKU:")
search_type = st.selectbox("What kind of match do you want?", ["GE only", "Competitor (non-GE)"])

# --- RUN MATCH AND DISPLAY ---
if input_sku:
    result_df = find_matches(input_sku, brand_filter="ge" if search_type == "GE only" else "non-ge")

    if isinstance(result_df, pd.DataFrame):
        result_df = result_df.reset_index(drop=True)

        # Show Competitor Info
        brand_col = 'Brand' if 'Brand' in df.columns else 'spec_14'
        config_col = 'Configuration' if 'Configuration' in df.columns else 'spec_7'
        status_col = 'Model Status' if 'Model Status' in df.columns else 'spec_9'
        description_col = 'Description' if 'Description' in df.columns else None

        competitor_row = df[df['SKU'] == input_sku]
        if not competitor_row.empty:
            competitor_data = {
                "SKU": input_sku,
                "Brand": competitor_row.iloc[0].get(brand_col, ''),
                "Configuration": competitor_row.iloc[0].get(config_col, ''),
                "Model Status": competitor_row.iloc[0].get(status_col, '')
            }
            if description_col:
                competitor_data["Description"] = str(competitor_row.iloc[0][description_col]).strip()

            competitor_df = pd.DataFrame([competitor_data])
            st.subheader("📦 Competitor SKU Details")
            st.table(competitor_df.astype(str))

        # Show Match Results
        st.subheader("📊 Closest Matching SKUs")
        safe_dicts = [{k: str(v) for k, v in row.items()} for _, row in result_df.iterrows()]
        cleaned_df = pd.DataFrame(safe_dicts)
        st.table(cleaned_df)

    elif isinstance(result_df, str):
        st.warning(result_df)
    else:
        st.error("Unexpected result format.")
