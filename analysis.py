import streamlit as st


def dataset_summary(df):

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.write("### Data Types")
    st.write(df.dtypes)

    st.write("### Statistical Summary")
    st.write(df.describe())
