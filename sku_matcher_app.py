import streamlit as st

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

# Step 2: Load and transpose the single sheet
import pandas as pd

df_raw = pd.read_excel(uploaded_file, header=None)

# Transpose so columns (SKUs) become rows
df = df_raw.T

# Set the first row as column headers
df.columns = df.iloc[0]
df = df[1:]  # Drop the header row
df.reset_index(drop=True, inplace=True)

# Rename SKU column
if 'SKU' not in df.columns:
    df.rename(columns={df.columns[0]: 'SKU'}, inplace=True)

# Fill missing values and ensure SKU is string
df.fillna('', inplace=True)
df.infer_objects(copy=False)
df['SKU'] = df['SKU'].astype(str)

# Step 3: Combine spec fields into one string
spec_columns = [col for col in df.columns if col != 'SKU']
df['combined_specs'] = df[spec_columns].astype(str).agg(' '.join, axis=1)
df = df.copy()


# Step 4: Build similarity model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_specs'])


# Step 5a: Find similar GE SKUs with same configuration
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

    # Build columns to return
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

# Step 5b: Find similar SKUs from all brands EXCEPT GE with same configuration
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

    # Build columns to return
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


# 📥 Ask the user for a SKU
input_sku = st.text_input("Enter a competitor SKU:")

# ✨ Optional: Choose search type
search_type = st.selectbox("What kind of match do you want?", ["GE only", "Competitor (non-GE)"])

# 🔍 Trigger matching only when input is provided
if input_sku:
    if search_type == "GE only":
        result_df = find_similar_ge_same_config(input_sku)
    else:
        result_df = find_similar_non_ge_same_config(input_sku)

import json  # make sure this is at the top of your script if not already

def is_displayable(df):
    try:
        json.dumps(df.to_dict(orient="records"))  # simulate Streamlit rendering
        return True
    except Exception as e:
        st.error(f"DataFrame serialization failed: {e}")
        return False

st.write("🧪 result_df type:", type(result_df))
if isinstance(result_df, pd.DataFrame):
    st.write("✅ result_df columns + types:")
    st.write(result_df.dtypes)


    result_df = result_df.copy()
    result_df = result_df.reset_index(drop=True)
    result_df = result_df.astype(str)
    result_df.columns = result_df.columns.astype(str)
    st.write("📌 result_df shape:", result_df.shape)
    st.write("📌 result_df preview:")
    st.write(result_df.head())
    st.write("📌 result_df columns:", result_df.columns.tolist())
    st.write("📌 result_df dtypes:", result_df.dtypes.to_dict())

    if is_displayable(result_df):
        st.dataframe(result_df)
else:
    st.error(result_df)
