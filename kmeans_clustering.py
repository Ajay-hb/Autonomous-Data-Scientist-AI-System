import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st

def perform_kmeans_clustering(df, n_clusters=3):

    # Select only numeric columns for clustering
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.empty:
        st.warning("No numeric columns available for clustering.")
        return None, None, None

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

    # Perform K-Means
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_df)
        df['Cluster'] = clusters

        # Calculate Silhouette Score if more than one cluster
        if n_clusters > 1:
            score = silhouette_score(scaled_df, clusters)
        else:
            score = None

        return df, score, numeric_df.columns.tolist()
    except Exception as e:
        st.error(f"Error during K-Means clustering: {e}")
        return None, None, None
