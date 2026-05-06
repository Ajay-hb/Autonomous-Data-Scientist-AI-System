import pandas as pd


def clean_data(df):

    # Remove duplicates
    df = df.drop_duplicates()

    # Numerical columns
    num_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    # Fill missing numerical values
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Categorical columns
    cat_cols = df.select_dtypes(
        include=['object']
    ).columns

    # Fill missing categorical values
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df
