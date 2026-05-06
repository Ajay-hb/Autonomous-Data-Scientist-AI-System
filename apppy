import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype # Import for type detection
import sys
import os
import matplotlib.pyplot as plt
import io
import importlib # Import importlib
import plotly.express as px # Added this import

# Add the project's root directory ('autods-ai') to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'autods-ai')))

# --- Start of fix for module caching ---
# If the module is already loaded (cached), remove it to force a reload from disk
if 'src.visualization.charts' in sys.modules:
    del sys.modules['src.visualization.charts']
if 'src.modeling.hyperparameter_tuning' in sys.modules:
    del sys.modules['src.modeling.hyperparameter_tuning']
if 'src.reports.pdf_generator' in sys.modules:
    del sys.modules['src.reports.pdf_generator']
# --- End of fix ---

from cleaning import clean_data
from analysis import dataset_summary
from charts import plot_histogram, get_matplotlib_seaborn_plot
from train_model import train_model
from kmeans_clustering import perform_kmeans_clustering
from dbscan_clustering import perform_dbscan_clustering
from helpers import sanitize_col_names
from pdf_generator import generate_pdf_report # Import the new PDF generator

st.set_page_config(
    page_title="AutoDS AI",
    layout="wide"
)

st.title("🤖 AutoDS AI")
st.subheader("Autonomous Data Scientist System")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

# Initialize session state for plots if not already present
if 'generated_plots' not in st.session_state:
    st.session_state.generated_plots = []
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = {}
if 'df_summary' not in st.session_state:
    st.session_state.df_summary = {}


if uploaded_file:

    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset Loaded Successfully")

    st.write("### Raw Dataset")
    st.dataframe(df.head())

    # Capture raw data head for PDF report
    st.session_state.df_summary['Raw Data Head'] = df.head().to_html(index=False)

    # Cleaning
    df = clean_data(df)

    # Sanitize column names right after cleaning
    df = sanitize_col_names(df)

    st.success("Dataset Cleaned and Column Names Sanitized")

    # --- Debugging Info ---
    st.write("--- Debugging Info ---")
    st.write("Data Types after cleaning:")
    st.write(df.dtypes)
    numeric_cols = df.select_dtypes(include='number').columns
    st.write(f"Numeric Columns detected: {', '.join(numeric_cols)}")
    st.write("------------------------")

    # Dataset Summary
    st.write("## Dataset Summary")
    # Capture summary for PDF
    st.session_state.df_summary.update({
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum()),
        "Data Types": df.dtypes.to_frame('Data Type').to_html() # Convert dtypes to a DataFrame then to HTML for report
    })
    # Capture statistical summary for PDF
    st.session_state.df_summary['Statistical Summary'] = df.describe().to_html()
    dataset_summary(df)

    # Visualization
    st.write("## Visualization")

    # Always show Plotly Histogram for numeric columns if available
    if len(numeric_cols) > 0:
        st.write("### Plotly Histogram (Single Column)")
        selected_col_plotly = st.selectbox(
            "Select Column for Plotly Histogram",
            numeric_cols,
            key="plotly_hist_select"
        )
        if selected_col_plotly:
            fig_plotly = plot_histogram(df, selected_col_plotly)
            st.plotly_chart(fig_plotly)

    st.write("### Advanced Matplotlib/Seaborn Visualizations")

    plot_name_input = st.text_input(
        "Enter a name for your plot (e.g., 'Age Distribution', 'Feature Correlation')",
        key="plot_name_input"
    )

    all_columns = df.columns.tolist()

    # X-axis selection
    x_axis_col = st.selectbox(
        "Select X-axis Column (primary for single-variable plots like Histograms, Box Plots, Density, QQ; first for Scatter/Overlaid)",
        ['None'] + all_columns,
        key="x_axis_select"
    )

    # Y-axis selection (only really applicable for Scatter, Overlaid Histograms)
    y_axis_col = st.selectbox(
        "Select Y-axis Column (for plots like Scatter, Overlaid Histograms)",
        ['None'] + all_columns,
        key="y_axis_select"
    )

    # General selection for multi-column plots (Pair Plot, Heatmap) or when no X/Y is given
    selected_cols_general = st.multiselect(
        "Select additional columns (for multi-variable plots like Pair Plot/Heatmap or when X/Y are 'None')",
        all_columns,
        default=all_columns[:3] if len(all_columns) > 0 else [],
        key="selected_cols_general_viz"
    )


    plot_type_advanced = st.selectbox(
        "Select Plot Type",
        ('Histogram', 'Box Plot', 'Scatter Plot', 'Pair Plot', 'Correlation Heatmap', 'Pie Chart', 'Density Plot', 'QQ Plot', 'Overlaid Histograms'),
        key="plot_type_type"
    )

    if st.button("Generate Plot"):
        if not plot_name_input:
            st.warning("Please enter a name for your plot before generating.")
        else:
            # Pass all relevant selections to the plotting function
            generated_fig = get_matplotlib_seaborn_plot(
                df,
                plot_type_advanced,
                selected_cols=selected_cols_general,
                x_col=x_axis_col,
                y_col=y_axis_col
            )
            if generated_fig:
                st.session_state.generated_plots.append({
                    'name': plot_name_input,
                    'figure': generated_fig,
                    'type': 'matplotlib'
                })
                st.success(f"Plot '{plot_name_input}' generated and added.")

    if st.session_state.generated_plots:
        st.write("### Your Generated Plots")
        for i, plot_data in enumerate(st.session_state.generated_plots):
            st.subheader(f"{i+1}. {plot_data['name']}")
            if plot_data['type'] == 'matplotlib':
                st.pyplot(plot_data['figure'])
                # Save button for Matplotlib
                buf = io.BytesIO()
                plot_data['figure'].savefig(buf, format="png", bbox_inches="tight")
                st.download_button(
                    label=f"Download {plot_data['name']} (PNG)",
                    data=buf.getvalue(),
                    file_name=f"{plot_data['name'].replace(' ', '_').lower()}.png",
                    mime="image/png",
                    key=f"download_mpl_{i}"
                )
                plt.close(plot_data['figure']) # Close figure to free memory
            else: # Plotly (though current implementation only adds matplotlib figs)
                st.plotly_chart(plot_data['figure'])
                # Plotly figures can also be saved, but requires more specific handling or client-side save

        if st.button("Clear All Generated Plots"):
            st.session_state.generated_plots = []
            st.rerun()

    # ML Section
    st.write("## Machine Learning")

    target_col = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # Determine problem type based on target column
    problem_type = ""
    if target_col and target_col in df.columns:
        if is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20:
            problem_type = "regression"
            st.info(f"Detected problem type: Regression (Target column '{target_col}' is numeric with many unique values).")
        else:
            problem_type = "classification"
            st.info(f"Detected problem type: Classification (Target column '{target_col}' is categorical or numeric with few unique values).")

    # Define ALL available models for classification and regression
    classification_models = (
        "Random Forest", "Logistic Regression", "XGBoost", "CatBoost",
        "LightGBM", "Naive Bayes", "SVM"
    )
    regression_models = (
        "Random Forest", "Linear Regression", "Ridge Regression", "Lasso Regression",
        "XGBoost", "CatBoost", "LightGBM", "SVM"
    )

    # Filter models based on detected problem type
    if problem_type == "regression":
        available_models = regression_models
    elif problem_type == "classification":
        available_models = classification_models
    else:
        available_models = () # Should not happen if target_col is selected

    # Display relevant models in a selectbox
    model_selection = st.selectbox(
        "Select Model Type",
        available_models
    )

    if st.button("Train Model", key="train_model_main"):
        if model_selection:
            metric_value, metric_name = train_model(df, target_col, model_selection)

            if metric_value is not None:
                st.success(f"Model {metric_name}: {metric_value:.2f}")
                st.session_state.model_results = {
                    "model_type": problem_type,
                    "selected_model_name": model_selection,
                    "target_col": target_col,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "tuning_enabled": False
                }
            else:
                st.error("Model training failed.")
        else:
            st.warning("Cannot train model without a selected model type.")

    # Hyperparameter Tuning Section
    st.write("## Hyperparameter Tuning")

    enable_tuning = st.checkbox("Enable Hyperparameter Tuning")

    if enable_tuning:
        tuning_method = st.selectbox(
            "Select Tuning Method",
            ("Optuna", "GridSearchCV", "RandomizedSearchCV")
        )

        n_tuning_trials = None # Initialize for Optuna
        n_iter_random = None   # Initialize for RandomizedSearchCV

        if tuning_method == "Optuna":
            n_tuning_trials = st.slider(
                "Number of Optuna Trials",
                min_value=5,
                max_value=100,
                value=20,
                step=5
            )
        elif tuning_method == "RandomizedSearchCV":
            n_iter_random = st.slider(
                "Number of Random Search Iterations",
                min_value=5,
                max_value=100,
                value=10,
                step=1
            )

        cv_folds = st.slider(
            "Cross-validation folds (CV)",
            min_value=2,
            max_value=10,
            value=5,
            step=1
        )

        if st.button("Train Model with Tuning", key="train_model_with_tuning"):
            if model_selection:
                from hyperparameter_tuning import optimize_hyperparameters
                with st.spinner(f"Running {tuning_method} Hyperparameter Tuning..."):
                    metric_value, metric_name, best_params, best_model = optimize_hyperparameters(
                        df, target_col, model_selection, problem_type,
                        tuning_method=tuning_method,
                        n_trials=n_tuning_trials,
                        n_iter_random=n_iter_random,
                        cv_folds=cv_folds
                    )

                if metric_value is not None:
                    st.success(f"Hyperparameter Tuning Complete! Best {metric_name}: {metric_value:.2f}")
                    st.write("Best Parameters found:")
                    st.json(best_params)
                    st.session_state.model_results = {
                        "model_type": problem_type,
                        "selected_model_name": model_selection,
                        "target_col": target_col,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "tuning_enabled": True,
                        "tuning_method": tuning_method,
                        "best_params": best_params
                    }
                else:
                    st.error(f"Hyperparameter tuning with {tuning_method} failed. Check console for details.")
            else:
                st.warning("Cannot train model without a selected model type.")

    # Unsupervised Learning Section
    st.write("## Unsupervised Learning - Clustering")

    st.markdown("Perform K-Means clustering on selected numeric features.")

    # Use the same numeric_cols for clustering selection
    clustering_cols = st.multiselect(
        "Select features for K-Means Clustering",
        numeric_cols,
        default=numeric_cols.tolist() if not numeric_cols.empty else [],
        key="clustering_cols_select"
    )

    n_clusters = st.slider(
        "Select Number of Clusters (K)",
        min_value=2,
        max_value=10,
        value=3,
        key="n_clusters_slider"
    )

    if st.button("Perform K-Means Clustering", key="kmeans_button"):
        if not clustering_cols:
            st.warning("Please select at least one feature for clustering.")
        else:
            # Create a dataframe with only selected clustering columns
            df_for_clustering = df[clustering_cols]
            clustered_df, silhouette_val, feature_names = perform_kmeans_clustering(df_for_clustering, n_clusters)

            if clustered_df is not None:
                st.success(f"K-Means Clustering Completed with {n_clusters} clusters.")
                st.write("### Clustered Data (first 5 rows with new 'Cluster' column)")
                st.dataframe(clustered_df.head())

                if silhouette_val is not None:
                    st.info(f"Silhouette Score: {silhouette_val:.2f}")

                # Store K-Means results in session state
                st.session_state.clustering_results['K-Means'] = {
                    'features': clustering_cols,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette_val
                }

                # Optional: Display cluster distribution
                st.write("### Cluster Distribution")
                cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
                fig_dist = plt.figure(figsize=(8, 4))
                plt.bar(cluster_counts.index, cluster_counts.values)
                plt.title('Cluster Distribution')
                st.pyplot(fig_dist)
                st.session_state.clustering_results['K-Means']['distribution_plot'] = fig_dist
                plt.close(fig_dist) # Close to free memory

                # Optional: Visualize clusters (e.g., scatter plot of two features colored by cluster)
                if len(clustering_cols) >= 2:
                    st.write("### Cluster Visualization (first two selected features)")
                    fig_cluster = px.scatter(
                        clustered_df,
                        x=clustering_cols[0],
                        y=clustering_cols[1],
                        color='Cluster',
                        title=f'Clusters of {clustering_cols[0]} vs {clustering_cols[1]}'
                    )
                    st.plotly_chart(fig_cluster)
                    # Note: Plotly figures are harder to pass directly to reportlab without converting to static image
                    # For simplicity, we won't store plotly figs in session_state for PDF report.
                elif len(clustering_cols) == 1:
                    st.write("### Cluster Visualization (first selected feature)")
                    fig_cluster_hist = px.histogram(
                        clustered_df,
                        x=clustering_cols[0],
                        color='Cluster',
                        title=f'Clusters Distribution for {clustering_cols[0]}'
                    )
                    st.plotly_chart(fig_cluster_hist)
                else:
                    st.info("Select at least two features for an insightful cluster visualization.")


    # DBSCAN Clustering Section
    st.write("## Unsupervised Learning - DBSCAN Clustering")

    st.markdown("Perform DBSCAN clustering on selected numeric features.")

    # Use the same numeric_cols for DBSCAN selection
    dbscan_clustering_cols = st.multiselect(
        "Select features for DBSCAN Clustering",
        numeric_cols,
        default=numeric_cols.tolist() if not numeric_cols.empty else [],
        key="dbscan_clustering_cols_select"
    )

    eps_value = st.slider(
        "Select eps (maximum distance between two samples for one to be considered as in the neighborhood of the other)",
        min_value=0.1,
        max_value=5.0,
        value=0.5,
        step=0.1,
        key="eps_slider"
    )

    min_samples_value = st.slider(
        "Select min_samples (number of samples in a neighborhood for a point to be considered as a core point)",
        min_value=2,
        max_value=20,
        value=5,
        step=1,
        key="min_samples_slider"
    )

    if st.button("Perform DBSCAN Clustering", key="dbscan_button"):
        if not dbscan_clustering_cols:
            st.warning("Please select at least one feature for DBSCAN clustering.")
        else:
            # Create a dataframe with only selected clustering columns
            df_for_dbscan_clustering = df[dbscan_clustering_cols]
            dbscan_clustered_df, dbscan_silhouette_val, dbscan_feature_names = perform_dbscan_clustering(
                df_for_dbscan_clustering,
                eps=eps_value,
                min_samples=min_samples_value
            )

            if dbscan_clustered_df is not None:
                st.success("DBSCAN Clustering Completed.")
                st.write("### Clustered Data (first 5 rows with new 'Cluster' column)")
                st.dataframe(dbscan_clustered_df.head())

                unique_clusters = dbscan_clustered_df['Cluster'].nunique()
                noise_points = (dbscan_clustered_df['Cluster'] == -1).sum()

                st.info(f"Number of clusters found: {unique_clusters - (1 if -1 in dbscan_clustered_df['Cluster'].unique() else 0)}")
                st.info(f"Number of noise points: {noise_points}")

                if dbscan_silhouette_val is not None:
                    st.info(f"Silhouette Score (excluding noise): {dbscan_silhouette_val:.2f}")

                # Store DBSCAN results in session state
                st.session_state.clustering_results['DBSCAN'] = {
                    'features': dbscan_clustering_cols,
                    'eps': eps_value,
                    'min_samples': min_samples_value,
                    'silhouette_score': dbscan_silhouette_val,
                    'noise_points': noise_points
                }

                # Optional: Display cluster distribution
                st.write("### DBSCAN Cluster Distribution")
                dbscan_cluster_counts = dbscan_clustered_df['Cluster'].value_counts().sort_index()
                fig_dbscan_dist = plt.figure(figsize=(8, 4))
                plt.bar(dbscan_cluster_counts.index, dbscan_cluster_counts.values)
                plt.title('DBSCAN Cluster Distribution')
                st.pyplot(fig_dbscan_dist)
                st.session_state.clustering_results['DBSCAN']['distribution_plot'] = fig_dbscan_dist
                plt.close(fig_dbscan_dist)

                # Optional: Visualize clusters (e.g., scatter plot of two features colored by cluster)
                if len(dbscan_clustering_cols) >= 2:
                    st.write("### DBSCAN Cluster Visualization (first two selected features)")
                    fig_dbscan_cluster = px.scatter(
                        dbscan_clustered_df,
                        x=dbscan_clustering_cols[0],
                        y=dbscan_clustering_cols[1],
                        color='Cluster',
                        title=f'DBSCAN Clusters of {dbscan_clustering_cols[0]} vs {dbscan_clustering_cols[1]}'
                    )
                    st.plotly_chart(fig_dbscan_cluster)
                elif len(dbscan_clustering_cols) == 1:
                    st.write("### DBSCAN Cluster Visualization (first selected feature)")
                    fig_dbscan_cluster_hist = px.histogram(
                        dbscan_clustered_df,
                        x=dbscan_clustering_cols[0],
                        color='Cluster',
                        title=f'DBSCAN Clusters Distribution for {dbscan_clustering_cols[0]}'
                    )
                    st.plotly_chart(fig_dbscan_cluster_hist)
                else:
                    st.info("Select at least two features for an insightful DBSCAN cluster visualization.")


    st.write("## Generate Report")
    if st.button("Generate PDF Report"):
        if not uploaded_file:
            st.warning("Please upload a dataset first to generate a report.")
        else:
            with st.spinner("Generating PDF report..."):
                pdf_output = generate_pdf_report(
                    st.session_state.df_summary,
                    st.session_state.generated_plots,
                    st.session_state.model_results,
                    st.session_state.clustering_results
                )
            st.download_button(
                label="Download PDF Report",
                data=pdf_output,
                file_name="autods_ai_report.pdf",
                mime="application/pdf"
            )
            st.success("PDF Report Generated!")
