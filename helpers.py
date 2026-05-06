import re

def sanitize_col_names(df):
    """Sanitizes DataFrame column names to be compatible with various ML models and Streamlit.
    Replaces problematic characters with underscores and removes leading/trailing underscores.
    """
    new_cols = []
    for col in df.columns:
        # Convert column name to string to handle cases where it might not be (e.g., multi-index after some ops)
        clean_col = re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) # Replace non-alphanumeric (except _) with _
        clean_col = clean_col.strip('_') # Remove leading/trailing underscores
        new_cols.append(clean_col)
    df.columns = new_cols
    return df
