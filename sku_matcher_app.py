import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 🔐 PASSWORD PROTECTION
PASSWORD = "geonly123"
pwd = st.text_input("Enter password", type="password")
if pwd != PASSWORD:
    st.warning("Access denied.")
    st.stop()

# 📁 FILE UPLOADER
st.title("GE SKU Matching Tool")
uploaded_file = st.file_uploader("Upload your SKU Excel file", type=["xlsx", "xls"])
if not uploaded_file:
    st.stop()

# 🧼 Load and preprocess Excel file
df_raw = pd.read_excel(uploaded_file, header=None)
df = df_raw.T
df.columns = df.iloc[0]
df = df[1:]
df.reset_index(drop=True, inplace=True)

# 🔍 Look for a row that contains description-like values
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

# 🔍 Basic cleaning
if 'SKU' not in df.columns:
    df.rename(columns={df.columns[0]: 'SKU'}, inplace=True)
df.fillna('', inplace=True)
df['SKU'] = df['SKU'].astype(str)

# 🔀 Build combined spec string for similarity
spec_columns = [col for col in df.columns if col != 'SKU']
df['combined_specs'] = df[spec_columns].astype(str).agg(' '.join, axis=1)

# 🔢 TF-IDF Model
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_specs'])

# 🔎 Matching Function
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
    if description_col:
        columns_to_return.append(description_col)
    columns_to_return += [config_col, status_col]

    rename_dict = {
        brand_col: 'Brand',
        config_col: 'Configuration',
        status_col: 'Model Status',
    }
    if description_col:
        rename_dict[description_col] = 'Description'

    return filtered[columns_to_return].rename(columns=rename_dict).head(top_n)

# 🧠 User Input
input_sku = st.text_input("Enter a competitor SKU:")
search_type = st.selectbox("What kind of match do you want?", ["GE only", "Competitor (non-GE)"])

# 🖥️ Show Matches
if input_sku:
    result_df = find_matches(input_sku, brand_filter="ge" if search_type == "GE only" else "non-ge")

    if isinstance(result_df, pd.DataFrame):
        result_df = result_df.reset_index(drop=True)

        # Competitor Row
        competitor_row = df[df['SKU'] == input_sku]
        if not competitor_row.empty:
            brand_col = 'Brand' if 'Brand' in df.columns else 'spec_14'
            config_col = 'Configuration' if 'Configuration' in df.columns else 'spec_7'
            status_col = 'Model Status' if 'Model Status' in df.columns else 'spec_9'
            description_col = 'Description' if 'Description' in df.columns else None

            competitor_data = {
                "SKU": input_sku,
                "Brand": competitor_row.iloc[0].get(brand_col, ''),
                "Configuration": competitor_row.iloc[0].get(config_col, ''),
                "Model Status": competitor_row.iloc[0].get(status_col, '')
            }
            if description_col:
                raw_desc = competitor_row.iloc[0][description_col]
                cleaned_desc = re.sub(r"^\d+\s*Description\s*", "", str(raw_desc), flags=re.IGNORECASE)
                competitor_data["Description"] = cleaned_desc

            ordered_cols = ['SKU', 'Brand']
            if 'Description' in competitor_data: ordered_cols.append('Description')
            ordered_cols += ['Configuration', 'Model Status']

            competitor_df = pd.DataFrame([competitor_data])[ordered_cols].astype(str)
            competitor_df.index = ['Competitor']

            result_df = result_df[ordered_cols].astype(str)
            final_display_df = pd.concat([competitor_df, result_df])

            st.subheader("📊 Comparison: Competitor vs. Closest GE Matches")
            st.table(final_display_df)

    elif isinstance(result_df, str):
        st.warning(result_df)
    else:
        st.error("Unexpected result format.")
