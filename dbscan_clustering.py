import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st

def perform_dbscan_clustering(df, eps=0.5, min_samples=5):

    # Select only numeric columns for clustering
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.empty:
        st.warning("No numeric columns available for DBSCAN clustering.")
        return None, None, None

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

    # Perform DBSCAN
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_df)

        # Add cluster assignments to the original (or selected features) DataFrame
        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters

        # Calculate Silhouette Score if more than one cluster and not all noise
        if len(set(clusters)) > 1 and -1 in clusters and len(set(clusters)) > 2: # -1 is noise
            # Exclude noise points for silhouette score calculation
            silhouette_samples = scaled_df[clusters != -1]
            silhouette_clusters = clusters[clusters != -1]
            if len(set(silhouette_clusters)) > 1:
                score = silhouette_score(silhouette_samples, silhouette_clusters)
            else:
                score = None # Only one cluster after removing noise
        elif len(set(clusters)) > 1 and -1 not in clusters:
            score = silhouette_score(scaled_df, clusters)
        else:
            score = None # Only one cluster or all noise

        return df_clustered, score, numeric_df.columns.tolist()
    except Exception as e:
        st.error(f"Error during DBSCAN clustering: {e}")
        return None, None, None
