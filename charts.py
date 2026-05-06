import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import math
import io
import scipy.stats as stats
import statsmodels.api as sm

def plot_histogram(df, column):
    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}"
    )
    return fig

def get_matplotlib_seaborn_plot(df, plot_type, x_col=None, y_col=None, selected_cols=None):
    """
    Generates a Matplotlib/Seaborn plot and returns the figure object.
    Supports flexible column selection via x_col, y_col, or selected_cols.
    """
    fig = None

    # Normalize 'None' string from selectbox to actual None
    x_col_effective = x_col if x_col != 'None' else None
    y_col_effective = y_col if y_col != 'None' else None

    # --- Handle Multi-Variable Plots (Pair Plot, Correlation Heatmap) ---
    if plot_type in ['Pair Plot', 'Correlation Heatmap']:
        cols_for_multi_plot = []
        if selected_cols:
            cols_for_multi_plot = [col for col in selected_cols if pd.api.types.is_numeric_dtype(df[col])]
        else: # Fallback to all numeric if no specific selected_cols
            cols_for_multi_plot = df.select_dtypes(include='number').columns.tolist()

        if len(cols_for_multi_plot) < 2:
            st.warning(f"{plot_type} requires at least two numeric columns. Please select them or ensure numeric columns exist.")
            return None

        if plot_type == 'Pair Plot':
            pair_grid = sns.pairplot(df[cols_for_multi_plot])
            fig = pair_grid.fig
        elif plot_type == 'Correlation Heatmap':
            corr_matrix = df[cols_for_multi_plot].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title('Correlation Heatmap')

        return fig # Return early as multi-variable plots are handled

    # --- Handle Single-Variable Plots ---
    if plot_type in ['Histogram', 'Box Plot', 'Density Plot', 'QQ Plot', 'Pie Chart']:
        if not x_col_effective:
            st.warning(f"Please select an X-axis column for {plot_type}.")
            return None

        is_x_numeric = pd.api.types.is_numeric_dtype(df[x_col_effective])

        if plot_type == 'Histogram':
            if is_x_numeric:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(df[x_col_effective], kde=True, ax=ax)
                ax.set_title(f'Histogram of {x_col_effective}')
            else:
                st.warning(f"Histogram requires a numeric X-axis column. '{x_col_effective}' is not numeric.")

        elif plot_type == 'Box Plot':
            if is_x_numeric:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(y=df[x_col_effective], ax=ax)
                ax.set_title(f'Box Plot of {x_col_effective}')
            else:
                st.warning(f"Box Plot requires a numeric X-axis column. '{x_col_effective}' is not numeric.")

        elif plot_type == 'Pie Chart':
            if not is_x_numeric: # Pie chart needs categorical/object
                fig, ax = plt.subplots(figsize=(8, 8))
                value_counts = df[x_col_effective].value_counts()
                if len(value_counts) > 10: # Limit to top 10 categories for readability
                    top_10_values = value_counts.nlargest(10)
                    other_sum = value_counts[~value_counts.index.isin(top_10_values.index)].sum()
                    value_counts = top_10_values
                    if other_sum > 0:
                        value_counts['Other'] = other_sum
                ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'Distribution of {x_col_effective}')
                ax.axis('equal')
            else:
                st.warning(f"Pie chart requires a categorical or object column. '{x_col_effective}' is numeric.")

        elif plot_type == 'Density Plot':
            if is_x_numeric:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.kdeplot(df[x_col_effective], fill=True, ax=ax)
                ax.set_title(f'Density Plot of {x_col_effective}')
            else:
                st.warning(f"Density Plot requires a numeric X-axis column. '{x_col_effective}' is not numeric.")

        elif plot_type == 'QQ Plot':
            if is_x_numeric:
                fig, ax = plt.subplots(figsize=(8, 5))
                sm.qqplot(df[x_col_effective], line='s', ax=ax)
                ax.set_title(f'QQ Plot of {x_col_effective}')
            else:
                st.warning(f"QQ Plot requires a numeric X-axis column. '{x_col_effective}' is not numeric.")
        return fig

    # --- Handle Two-Variable Plots ---
    if plot_type in ['Scatter Plot', 'Overlaid Histograms']:
        if not x_col_effective or not y_col_effective:
            st.warning(f"Please select both X and Y-axis columns for {plot_type}.")
            return None

        is_x_numeric = pd.api.types.is_numeric_dtype(df[x_col_effective])
        is_y_numeric = pd.api.types.is_numeric_dtype(df[y_col_effective])

        if plot_type == 'Scatter Plot':
            if is_x_numeric and is_y_numeric:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=df[x_col_effective], y=df[y_col_effective], ax=ax)
                ax.set_title(f'Scatter Plot of {x_col_effective} vs {y_col_effective}')
            else:
                st.warning("Scatter plot requires two numeric columns for X and Y axes.")

        elif plot_type == 'Overlaid Histograms':
            if is_x_numeric and is_y_numeric:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[x_col_effective], color="blue", label=x_col_effective, kde=True, alpha=0.5, ax=ax)
                sns.histplot(df[y_col_effective], color="red", label=y_col_effective, kde=True, alpha=0.5, ax=ax)
                ax.set_title(f'Overlaid Histograms of {x_col_effective} and {y_col_effective}')
                ax.legend()
            else:
                st.warning("Overlaid Histograms require two numeric columns for X and Y axes.")
        return fig

    st.error(f"Unsupported plot type: {plot_type}")
    return None
