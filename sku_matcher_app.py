import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

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

# Step 1: Load and transpose the Excel sheet
df_raw = pd.read_excel(uploaded_file, header=None)
df = df_raw.T
df.columns = df.iloc[0]
df = df[1:]
df.reset_index(drop=True, inplace=True)

# Ensure SKU column exists and clean
if 'SKU' not in df.columns:
    df.rename(columns={df.columns[0]: 'SKU'}, inplace=True)
df.fillna('', inplace=True)
df.infer_objects(copy=False)
df['SKU'] = df['SKU'].astype(str)

# Step 2: Combine all spec columns into one text field
spec_columns = [col for col in df.columns if col != 'SKU']
df['combined_specs'] = df[spec_columns].astype(str).agg(' '.join, axis=1)
df = df.copy()

# Step 3: TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_specs'])

# Step 4: Matching Functions

def find_similar_ge_same_config(input_sku, top_n=5):
    input_row = df[df['SKU'] == input_sku]
    if input_row.empty:
        return f"SKU {input_sku} not found."

    input_index = input_row.index[0]
    input_vector = tfidf_matrix[input_index]
    similarities = cosine_similarity(input_vector, tfidf_matrix)[0]

    brand_col = 'Brand' if 'Brand' in df.columns else 'spec_14'
    config_col = 'Configuration' if 'Configuration' in df.columns else 'spec_7'
    status_col = 'Model Status' if 'Model Status' in df.columns else 'spec_9'

    input_config = input_row.iloc[0][config_col]

    df_copy = df.copy()
    df_copy['similarity'] = similarities

    filtered = df_copy[
        (df_copy[brand_col].str.lower() == 'ge') &
        (df_copy[config_col].str.lower() == input_config.lower()) &
        (df_copy['SKU'] != input_sku)
    ].copy()

    filtered = filtered.sort_values(by='similarity', ascending=False)

    columns_to_return = ['SKU', brand_col, config_col, status_col]
    rename_dict = {
        brand_col: 'Brand',
        config_col: 'Configuration',
        status_col: 'Model Status'
    }

    if 'Console Control Type' in df.columns:
        columns_to_return.insert(-1, 'Console Control Type')
        rename_dict['Console Control Type'] = 'Console Control Type'
    elif 'spec_22' in df.columns:
        columns_to_return.insert(-1, 'spec_22')
        rename_dict['spec_22'] = 'Console Control Type'

    return filtered[columns_to_return].rename(columns=rename_dict).head(top_n)

def find_similar_non_ge_same_config(input_sku, top_n=5):
    input_row = df[df['SKU'] == input_sku]
    if input_row.empty:
        return f"SKU {input_sku} not found."

    input_index = input_row.index[0]
    input_vector = tfidf_matrix[input_index]
    similarities = cosine_similarity(input_vector, tfidf_matrix)[0]

    brand_col = 'Brand' if 'Brand' in df.columns else 'spec_14'
    config_col = 'Configuration' if 'Configuration' in df.columns else 'spec_7'
    status_col = 'Model Status' if 'Model Status' in df.columns else 'spec_9'

    input_config = input_row.iloc[0][config_col]

    df_copy = df.copy()
    df_copy['similarity'] = similarities

    filtered = df_copy[
        (df_copy[brand_col].str.lower() != 'ge') &
        (df_copy[config_col].str.lower() == input_config.lower()) &
        (df_copy['SKU'] != input_sku)
    ].copy()

    filtered = filtered.sort_values(by='similarity', ascending=False)

    columns_to_return = ['SKU', brand_col, config_col, status_col, 'similarity']
    rename_dict = {
        brand_col: 'Brand',
        config_col: 'Configuration',
        status_col: 'Model Status'
    }

    if 'Console Control Type' in df.columns:
        columns_to_return.insert(-1, 'Console Control Type')
        rename_dict['Console Control Type'] = 'Console Control Type'
    elif 'spec_22' in df.columns:
        columns_to_return.insert(-1, 'spec_22')
        rename_dict['spec_22'] = 'Console Control Type'

    return filtered[columns_to_return].rename(columns=rename_dict).head(top_n)

# Step 5: UI Inputs
input_sku = st.text_input("Enter a competitor SKU:")
search_type = st.selectbox("What kind of match do you want?", ["GE only", "Competitor (non-GE)"])

# Step 6: Execute Matching and Display Results
result_df = None

if input_sku:
    if search_type == "GE only":
        result_df = find_similar_ge_same_config(input_sku)
    else:
        result_df = find_similar_non_ge_same_config(input_sku)

    if isinstance(result_df, pd.DataFrame):
        result_df = result_df.copy()
        result_df = result_df.reset_index(drop=True).astype(str)
        result_df.columns = result_df.columns.astype(str)
        result_df = result_df.applymap(str)

        try:
            json.dumps(result_df.to_dict(orient="records"))  # test for displayability
            st.dataframe(result_df)
        except Exception as e:
            st.error(f"DataFrame serialization failed: {e}")
    elif isinstance(result_df, str):
        st.warning(result_df)
