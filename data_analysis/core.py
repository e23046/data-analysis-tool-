"""
data_analysis/core.py

Two main classes:
  - DataInspector  : load, clean, explore, and export tabular data.
  - PlottingMethods: standalone chart generators that return HTML-ready Plotly figures.
"""

from __future__ import annotations

import base64
import inspect
import io
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scipy
import scipy.stats
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, f_oneway, pointbiserialr
from sklearn.preprocessing import (
    MinMaxScaler, OrdinalEncoder, OneHotEncoder,
    RobustScaler, StandardScaler,
)


# ---------------------------------------------------------------------------
# DataInspector
# ---------------------------------------------------------------------------

class DataInspector:
    """
    A straightforward toolkit for cleaning and exploring tabular data.
    Works in Google Colab, standard Jupyter Notebooks, and plain Python scripts.
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.numeric_df: Optional[pd.DataFrame] = None
        self.categorical_df: Optional[pd.DataFrame] = None
        self.categorical_normalized_df: Optional[pd.DataFrame] = None
        self.normalized_data_df: Optional[pd.DataFrame] = None
        self.numeric_normalized_df: Optional[pd.DataFrame] = None
        self.plotting_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def upload_forced_file(self) -> None:
        """
        Repeatedly prompts the user via Google Colab file upload utilities 
        until a valid CSV data file is successfully selected and processed.
        """
        import io
        try:
            from google.colab import files as colab_files
        except ImportError:
            print("❌ This feature requires a Google Colab notebook environment.")
            return

        print("📢 System is waiting for a valid data submission...")
        
        while True:
            try:
                uploaded = colab_files.upload()
                
                # Check if the user closed the upload window without picking a file
                if not uploaded:
                    print("⚠️ No file chosen. You must upload a valid dataset file to proceed.\n")
                    continue
                
                name = list(uploaded.keys())[0]
                
                # --- STRATEGIC FORMAT MATCHING RULES ---
                if not name.lower().endswith('.csv'):
                    print(f"❌ Rejected: '{name}' is not a valid document layout.")
                    print("Please choose a file ending with the '.csv' format extension.\n")
                    continue
                
                # Directly match the loading properties of your original upload_data method
                self.df = pd.read_csv(
                    io.BytesIO(uploaded[name]),
                    na_values=["?", "n/a", "N/A", "NULL", "null", " "],
                )
                self.df["count"] = 1

                # Convert columns to numeric profiles where applicable
                for col in self.df.columns:
                    converted = pd.to_numeric(self.df[col], errors="coerce")
                    if not converted.isna().all():
                        self.df[col] = converted
                
                print(f"\n✅ Success: Verified file '{name}' uploaded successfully! Shape: {self.df.shape}")
                break  # 👈 Breaks out of the loop cleanly once data is processed successfully!
                
            except Exception as loop_error:
                print(f"❌ Structural read error: {str(loop_error)}")
                print("Please verify your data layout constraints and try again.\n")
              
    def upload_data(self):
        """
        Opens a file-upload dialog (Google Colab), reads the chosen CSV,
        and converts columns to numeric types wherever possible.
        Common null placeholders like '?', 'NULL', and 'N/A' are treated as NaN.
        """
        from google.colab import files as colab_files
        uploaded = colab_files.upload()

        if not uploaded:
            print("No file was uploaded.")
            return

        name = list(uploaded.keys())[0]
        self.df = pd.read_csv(
            io.BytesIO(uploaded[name]),
            na_values=["?", "n/a", "N/A", "NULL", "null", " "],
        )
        self.df["count"] = 1

        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors="coerce")
            if not converted.isna().all():
                self.df[col] = converted

        print(f"✅ '{name}' loaded successfully. Shape: {self.df.shape}")

    # ------------------------------------------------------------------
    # INSPECTION
    # ------------------------------------------------------------------

    def get_summary(self):
        """
        Prints the dataset shape, lists numeric and categorical columns,
        and displays the full DataFrame inline.
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        num_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != "count"]
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

        print("--- Summary ---")
        print(f"Rows: {self.df.shape[0]}  |  Columns: {self.df.shape[1]}")
        print(f"Numeric  ({len(num_cols)}): {num_cols}")
        print(f"Categorical ({len(cat_cols)}): {cat_cols}")

        from IPython.display import display
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(self.df)

    def column_details(self):
        """
        For each column: shows the value range (numeric) or the number of
        unique values (categorical).
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"  {col} [numeric]  min={self.df[col].min()}  max={self.df[col].max()}")
            else:
                print(f"  {col} [categorical]  {self.df[col].nunique()} unique values")

    def get_categorical_summary(self):
        """
        Shows a concise summary (unique count, most-frequent value, frequency)
        for every categorical column.
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.empty:
            print("No categorical columns found.")
            return

        from IPython.display import display
        display(cat_df.describe().T[["unique", "top", "freq"]])

    def show_missing_data(self):
        """
        Reports how many rows contain missing values.
        Optionally shows a True/False mask for a specific column.
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        missing_rows = self.df[self.df.isnull().any(axis=1) | (self.df == "").any(axis=1)]

        if missing_rows.empty:
            print("No missing values found.")
            return

        print(f"Found {len(missing_rows)} rows with at least one missing value.")

        choice = input("Show exact positions for a specific column? (yes/no): ").strip().lower()
        if choice in ("yes", "y"):
            print(f"Columns: {list(self.df.columns)}")
            col = input("Column name: ").strip()
            if col not in self.df.columns:
                print(f"Column '{col}' not found.")
                return
            mask = (self.df[col].isnull() | (self.df[col] == "")).map({True: "missing", False: "ok"})
            with pd.option_context("display.max_rows", None):
                print(mask.to_frame(name="status"))
        else:
            from IPython.display import display
            display(missing_rows.head(20))

    # ------------------------------------------------------------------
    # CLEANING
    # ------------------------------------------------------------------

    def handle_missing_values(self, strategy: str = "median", columns: Optional[List[str]] = None) -> None:
        """
        Safely imputes missing records. If a user uploads any arbitrary dataset,
        this will automatically filter out valid columns or fallback to processing 
        all columns so that it never crashes.
        """
        if self.df is None:
            print("❌ No dataset loaded yet.")
            return

        # 1. SMART AUTO-DETECTION: 
        # If columns are specified, only keep the ones that actually exist in the uploaded file.
        if columns:
            target_cols = [col for col in columns if col in self.df.columns]
            if not target_cols:
                print(f"⚠️ None of the requested columns {columns} were found. Falling back to all columns.")
                target_cols = list(self.df.columns)
        else:
            # If no columns are provided, automatically target all columns
            target_cols = list(self.df.columns)

        # Remove internal utility tracking columns if they exist
        target_cols = [c for c in target_cols if c != "count"]

        for col in target_cols:
            null_count = self.df[col].isnull().sum()
            if null_count == 0:
                continue

            if strategy == "drop":
                self.df = self.df.dropna(subset=[col])
            elif strategy == "mean" and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == "median" and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == "mode" or not pd.api.types.is_numeric_dtype(self.df[col]):
                mode_val = self.df[col].mode()
                if not mode_val.empty:
                    self.df[col] = self.df[col].fillna(mode_val[0])
                    
        print(f"✅ Successfully handled missing values using '{strategy}' strategy for valid columns.")

    def handle_outliers(self, find_and_delete: bool = True, strategy: str = "remove", columns: Optional[List[str]] = None) -> None:
        """
        Filters distribution outliers using robust Interquartile Range (IQR).
        Automatically skips non-existent or categorical columns to prevent errors on any dataset.
        """
        if self.df is None:
            print("❌ No dataset loaded yet.")
            return

        # 2. SMART AUTO-DETECTION FOR OUTLIERS:
        # Filter down to only columns that exist AND are numeric.
        if columns:
            target_cols = [col for col in columns if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])]
            if not target_cols:
                print("⚠️ No valid numeric columns found from input list. Auto-selecting all numeric features.")
                target_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        else:
            target_cols = list(self.df.select_dtypes(include=[np.number]).columns)

        target_cols = [c for c in target_cols if c != "count"]
        initial_shape = self.df.shape

        for col in target_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter rows dynamically
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                
        print(f"✅ Outliers processed. Rows before: {initial_shape[0]} -> Rows after: {self.df.shape[0]}")

    def remove_duplicates(self):
        """Drops exact duplicate rows and resets the index."""
        if self.df is None:
            print("No data loaded yet.")
            return

        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {before - len(self.df)} duplicate rows. Remaining: {len(self.df)}")

    

    def delete_rows(self, indices: Optional[List[int]] = None):
        """
        Deletes rows by index.
        Pass a list directly, or leave empty to type indices interactively.
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        if indices is None:
            raw = input("Row indices to delete (comma-separated): ")
            indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]

        valid = [i for i in indices if i in self.df.index]
        self.df = self.df.drop(index=valid).reset_index(drop=True)
        print(f"Deleted {len(valid)} rows. Remaining: {len(self.df)}")

    def delete_columns(self, columns: Optional[List[str]] = None):
        """
        Deletes columns by name.
        Pass a list directly, or leave empty to type column names interactively.
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        if columns is None:
            print(f"Available columns: {', '.join(self.df.columns)}")
            raw = input("Column names to delete (comma-separated): ")
            columns = [c.strip() for c in raw.split(",")]

        valid = [c for c in columns if c in self.df.columns]
        if not valid:
            print("None of the given column names were found.")
            return

        self.df = self.df.drop(columns=valid)
        print(f"Deleted {len(valid)} columns. Remaining: {len(self.df.columns)}")

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------

    def export_cleaned_data(self, filename: str = "cleaned_data.csv"):
        """
        Saves the current dataset to a CSV file.
        In Google Colab, a download is triggered automatically.
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        self.df.to_csv(filename, index=False)
        print(f"Saved to '{os.path.abspath(filename)}'")

        try:
            from google.colab import files as colab_files
            colab_files.download(filename)
        except ImportError:
            print("(Not running in Colab — file saved locally.)")

    # ------------------------------------------------------------------
    # SAMPLING
    # ------------------------------------------------------------------

    def sample_data(self, n: int = 20):
        """
        Randomly picks up to n rows for plotting. Uses the full dataset if it's smaller than n.
        """
        if self.df is None:
            print("No data loaded yet.")
            return None

        if len(self.df) <= n:
            self.plotting_df = self.df.copy()
            print(f"Dataset has {len(self.df)} rows — using all of them.")
        else:
            self.plotting_df = self.df.sample(n=n, random_state=42).reset_index(drop=True)
            print(f"Sampled {n} rows from {len(self.df)}.")

        return self.plotting_df

    def select_columns_for_plotting(self) -> List[str]:
        """
        Lists available columns and asks the user which ones to plot.
        Accepts column names, column numbers, or 'all'.
        """
        if self.df is None:
            print("No data loaded yet.")
            return []

        available = [c for c in self.df.columns if c != "count"]
        print("\nAvailable columns:")
        for i, col in enumerate(available, 1):
            print(f"  {i}. {col}")

        raw = input("Enter names or numbers (comma-separated), or type 'all': ").strip()
        if raw.lower() == "all":
            return available

        selected = []
        for item in [x.strip() for x in raw.split(",")]:
            if item.isdigit() and 0 < int(item) <= len(available):
                selected.append(available[int(item) - 1])
            elif item in available:
                selected.append(item)

        print(f"Selected: {selected}")
        return selected

    # ------------------------------------------------------------------
    # VISUALISATION — DISTRIBUTIONS
    # ------------------------------------------------------------------

    def plot_numerical(self, column_names: List[str]):
        """
        For each numeric column, shows a side-by-side violin/box, scatter, and histogram.
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        source = getattr(self, "plotting_df", None) or self.df
        valid = [
            c for c in column_names
            if c in source.columns and pd.api.types.is_numeric_dtype(source[c]) and c != "count"
        ]

        if not valid:
            print("No valid numeric columns found in the selection.")
            return

        for col in valid:
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=(f"Violin / Box: {col}", f"Scatter: {col}", f"Histogram: {col}"),
            )
            fig.add_trace(
                go.Violin(x=source[col], box_visible=True, meanline_visible=True,
                          name=col, orientation="h", line_color="lightseagreen"),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=list(range(len(source))), y=source[col],
                           mode="markers+lines", marker=dict(opacity=0.7, color="royalblue", size=6),
                           name=col),
                row=1, col=2,
            )
            fig.add_trace(
                go.Histogram(x=source[col], name=col, marker_color="indianred"),
                row=1, col=3,
            )
            fig.update_layout(
                height=380, width=1150,
                title_text=f"<b>Distribution Overview: {col}</b>",
                showlegend=False, template="plotly_white",
            )
            fig.show()

    def plot_categorical(self, column_names: List[str]):
        """
        For each categorical column, shows a bar chart with counts and percentages.
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        if isinstance(column_names, str):
            column_names = [column_names]

        source = getattr(self, "plotting_df", None) or self.df

        for col in [c for c in column_names if c in source.columns]:
            counts = source[col].value_counts().reset_index()
            counts.columns = [col, "count"]
            total = counts["count"].sum()
            counts["pct"] = ((counts["count"] / total) * 100).round(1)
            counts["label"] = counts.apply(lambda r: f"{r['count']} ({r['pct']}%)", axis=1)

            fig = px.bar(
                counts, x=col, y="count", text="label",
                title=f"<b>Frequency: {col}</b>",
                color=col, color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=400, template="plotly_white", showlegend=False)
            fig.show()

    def plot_relationship(self, col1: str, col2: str):
        """
        Picks the right chart type based on column types:
        - Numeric vs Numeric  → scatter with OLS trendline
        - Categorical vs Numeric → box plot
        - Categorical vs Categorical → grouped bar chart
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        source = getattr(self, "plotting_df", None) or self.df

        if col1 not in source.columns or col2 not in source.columns:
            print("One or both column names not found.")
            return

        is_num1 = pd.api.types.is_numeric_dtype(source[col1]) and col1 != "count"
        is_num2 = pd.api.types.is_numeric_dtype(source[col2]) and col2 != "count"

        if is_num1 and is_num2:
            if source[col1].nunique() <= 1 or source[col2].nunique() <= 1:
                fig = px.scatter(source, x=col1, y=col2,
                                 title=f"<b>Scatter: {col1} vs {col2}</b>")
            else:
                try:
                    import statsmodels  # noqa: F401
                    fig = px.scatter(source, x=col1, y=col2, trendline="ols",
                                     title=f"<b>Scatter + OLS Trendline: {col1} vs {col2}</b>")
                except ImportError:
                    print("Tip: install statsmodels to add a trendline.")
                    fig = px.scatter(source, x=col1, y=col2,
                                     title=f"<b>Scatter: {col1} vs {col2}</b>")

        elif is_num1 != is_num2:
            num, cat = (col1, col2) if is_num1 else (col2, col1)
            fig = px.box(source, x=cat, y=num, points="all", color=cat,
                         title=f"<b>{num} by {cat}</b>")

        else:
            grouped = source.groupby([col1, col2]).size().reset_index(name="count")
            fig = px.bar(grouped, x=col1, y="count", color=col2, barmode="group",
                         title=f"<b>{col1} vs {col2}</b>")

        fig.update_layout(template="plotly_white")
        fig.show()

    # ------------------------------------------------------------------
    # VISUALISATION — CORRELATIONS
    # ------------------------------------------------------------------

    def plot_numerical_correlation(self):
        """Displays a Pearson correlation heatmap for all numeric columns."""
        if self.df is None:
            print("No data loaded yet.")
            return

        num_df = self.df.select_dtypes(include=["number"]).drop(columns=["count"], errors="ignore")

        if num_df.shape[1] < 2:
            print("Need at least 2 numeric columns.")
            return

        variable_cols = num_df.loc[:, num_df.nunique() > 1]
        corr = pd.DataFrame(np.eye(num_df.shape[1]), index=num_df.columns, columns=num_df.columns)
        sub = variable_cols.corr(method="pearson")
        corr.loc[sub.index, sub.columns] = sub

        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="<b>Pearson Correlation Heatmap</b>",
        )
        fig.update_layout(width=700, height=600, template="plotly_white")
        fig.show()

    def plot_categorical_correlation(self):
        """Displays a Cramér's V heatmap for all categorical columns."""
        if self.df is None:
            print("No data loaded yet.")
            return

        cat_df = self.df.select_dtypes(exclude=["number"])
        if cat_df.shape[1] < 2:
            print("Need at least 2 categorical columns.")
            return

        cols = cat_df.columns
        n = len(cols)
        mat = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

        for i in range(n):
            for j in range(i, n):
                c1, c2 = cols[i], cols[j]
                if i == j:
                    mat.loc[c1, c2] = 1.0
                    continue
                ct = pd.crosstab(cat_df[c1], cat_df[c2])
                if ct.size > 0 and min(ct.shape) > 1:
                    chi2 = chi2_contingency(ct)[0]
                    total = ct.sum().sum()
                    v = np.sqrt(chi2 / (total * (min(ct.shape) - 1))) if total > 0 else 0.0
                    mat.loc[c1, c2] = mat.loc[c2, c1] = v

        fig = px.imshow(
            mat, text_auto=".2f", aspect="auto",
            color_continuous_scale="Viridis", zmin=0, zmax=1,
            title="<b>Cramér's V Association Heatmap</b>",
        )
        fig.update_layout(width=720, height=620, template="plotly_white")
        fig.show()

    def correlate_num_to_cat(self) -> pd.DataFrame:
        """
        Computes association strength between every numeric–categorical pair.
        Uses Point-Biserial for binary categories and Eta (ANOVA) for multi-class ones.
        """
        if self.df is None:
            print("No data loaded yet.")
            return pd.DataFrame()

        num_cols = [c for c in self.df.select_dtypes(include=["number"]).columns if c != "count"]
        cat_cols = self.df.select_dtypes(exclude=["number"]).columns.tolist()

        if not num_cols or not cat_cols:
            print("Dataset must have both numeric and categorical columns.")
            return pd.DataFrame()

        rows = []
        for num in num_cols:
            for cat in cat_cols:
                pair = self.df[[num, cat]].dropna()
                if pair.empty:
                    continue
                unique_cats = pair[cat].unique()
                if len(unique_cats) < 2:
                    continue

                if len(unique_cats) == 2:
                    try:
                        binary = (pair[cat] == unique_cats[0]).astype(int)
                        if binary.nunique() < 2 or pair[num].nunique() < 2:
                            val, method = 0.0, "Point-Biserial (constant)"
                        else:
                            r_pb, _ = pointbiserialr(binary, pair[num])
                            val = 0.0 if np.isnan(r_pb) else r_pb
                            method = "Point-Biserial"
                    except Exception:
                        val, method = 0.0, "Failed"
                else:
                    try:
                        groups = [pair[pair[cat] == c][num].values for c in unique_cats if len(pair[pair[cat] == c]) > 0]
                        if len(groups) > 1:
                            grand_mean = pair[num].mean()
                            ss_total = ((pair[num] - grand_mean) ** 2).sum()
                            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
                            val = np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
                            method = "Eta (ANOVA)"
                        else:
                            val, method = 0.0, "Insufficient groups"
                    except Exception:
                        val, method = 0.0, "Failed"

                rows.append({
                    "Numeric": num,
                    "Categorical": cat,
                    "Association": round(val, 4),
                    "Method": method,
                })

        result = pd.DataFrame(rows)
        from IPython.display import display
        display(result)
        return result

    def plot_all_associations_heatmap(self) -> pd.DataFrame:
        """
        Builds a unified association matrix covering all column pairs, regardless of type,
        and renders it as an interactive heatmap.
        """
        if self.df is None:
            print("No data loaded yet.")
            return pd.DataFrame()

        cols = [c for c in self.df.columns if c != "count"]
        n = len(cols)
        mat = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

        for i in range(n):
            for j in range(i, n):
                c1, c2 = cols[i], cols[j]
                if i == j:
                    mat.loc[c1, c2] = 1.0
                    continue

                pair = self.df[[c1, c2]].dropna()
                if pair.empty:
                    continue

                is_num1 = pd.api.types.is_numeric_dtype(pair[c1])
                is_num2 = pd.api.types.is_numeric_dtype(pair[c2])

                if is_num1 and is_num2:
                    val = abs(pair[c1].corr(pair[c2])) if pair[c1].nunique() > 1 and pair[c2].nunique() > 1 else 0.0

                elif not is_num1 and not is_num2:
                    ct = pd.crosstab(pair[c1], pair[c2])
                    if ct.size > 0 and min(ct.shape) > 1:
                        chi2 = chi2_contingency(ct)[0]
                        total = ct.sum().sum()
                        val = np.sqrt(chi2 / (total * (min(ct.shape) - 1))) if total > 0 else 0.0
                    else:
                        val = 0.0

                else:
                    cat_col, num_col = (c1, c2) if not is_num1 else (c2, c1)
                    cats = pair[cat_col].unique()
                    if len(cats) > 1:
                        groups = [pair[pair[cat_col] == c][num_col].values for c in cats if len(pair[pair[cat_col] == c]) > 0]
                        grand_mean = pair[num_col].mean()
                        ss_total = ((pair[num_col] - grand_mean) ** 2).sum()
                        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
                        val = np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
                    else:
                        val = 0.0

                mat.loc[c1, c2] = mat.loc[c2, c1] = round(val, 3)

        fig = px.imshow(
            mat, text_auto=".2f", aspect="auto",
            color_continuous_scale="Viridis", zmin=0, zmax=1,
            title="<b>Unified Association Heatmap (All Features)</b>",
            labels=dict(color="Strength"),
        )
        fig.update_layout(
            width=max(650, n * 50), height=max(550, n * 50), template="plotly_white"
        )
        fig.show()
        return mat

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------------------------

    def extract_numeric_data(self) -> Optional[pd.DataFrame]:
        """Returns a DataFrame with only the numeric columns."""
        if self.df is None:
            print("No data loaded yet.")
            return None
        self.numeric_df = self.df.select_dtypes(include=[np.number])
        return self.numeric_df

    def extract_categorical_data(self) -> Optional[pd.DataFrame]:
        """Returns a DataFrame with only the categorical columns."""
        if self.df is None:
            print("No data loaded yet.")
            return None
        self.categorical_df = self.df.select_dtypes(exclude=[np.number])
        return self.categorical_df

    def extract_normalized_numeric_data(self, method: str = "minmax") -> pd.DataFrame:
        """
        Scales numeric columns using 'minmax', 'standard', or 'robust'.
        Missing values are filled with column medians before scaling.
        """
        if self.df is None:
            print("No data loaded yet.")
            return pd.DataFrame()

        num_df = self.df.select_dtypes(include=["number"]).drop(columns=["count"], errors="ignore").copy()

        if num_df.empty:
            print("No numeric columns to scale.")
            self.numeric_normalized_df = pd.DataFrame()
            return self.numeric_normalized_df

        if num_df.isnull().any().any():
            num_df = num_df.fillna(num_df.median())

        scalers = {
            "minmax": MinMaxScaler(),
            "standard": StandardScaler(),
            "robust": RobustScaler(),
        }
        scaler = scalers.get(method.lower().strip())
        if scaler is None:
            print(f"Unknown method '{method}'. Using 'minmax'.")
            scaler = MinMaxScaler()

        scaled = scaler.fit_transform(num_df)
        self.numeric_normalized_df = pd.DataFrame(scaled, columns=num_df.columns, index=num_df.index)
        print(f"Numeric data scaled using '{method}'.")
        return self.numeric_normalized_df

    def extract_normalized_categorical_data(self, method: str = "uniform") -> pd.DataFrame:
        """
        Encodes categorical columns using 'uniform', 'ordinal', or 'onehot'.
        """
        if self.df is None:
            print("No data loaded yet.")
            return pd.DataFrame()

        cat_df = self.df.select_dtypes(exclude=["number"]).copy()
        if cat_df.empty:
            print("No categorical columns to encode.")
            return pd.DataFrame()

        filled = cat_df.fillna("Missing")
        m = method.lower().strip()

        if m == "uniform":
            self.categorical_normalized_df = pd.DataFrame(index=cat_df.index)
            for col in cat_df.columns:
                codes = cat_df[col].astype("category").cat.codes
                mx = codes.max()
                self.categorical_normalized_df[col] = codes / mx if mx > 0 else 0.0

        elif m == "ordinal":
            enc = OrdinalEncoder()
            encoded = enc.fit_transform(filled)
            self.categorical_normalized_df = pd.DataFrame(encoded, columns=cat_df.columns, index=cat_df.index)

        elif m == "onehot":
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = enc.fit_transform(filled)
            names = enc.get_feature_names_out(cat_df.columns)
            self.categorical_normalized_df = pd.DataFrame(encoded, columns=names, index=cat_df.index)

        else:
            print(f"Unknown encoding '{method}'. Falling back to 'uniform'.")
            return self.extract_normalized_categorical_data(method="uniform")

        print(f"Categorical data encoded using '{m}'.")
        return self.categorical_normalized_df

    def create_normalized_data_df(self) -> pd.DataFrame:
        """
        Merges the original numeric columns with the encoded categorical columns
        into a single ML-ready DataFrame.
        """
        if self.df is None:
            print("No data loaded yet.")
            return pd.DataFrame()

        num_orig = self.df.select_dtypes(include=["number"]).drop(columns=["count"], errors="ignore").copy()

        cat_norm = getattr(self, "categorical_normalized_df", None)
        if cat_norm is None:
            print("Running default uniform encoding on categorical columns...")
            cat_norm = self.extract_normalized_categorical_data(method="uniform")

        if cat_norm.empty:
            self.normalized_data_df = num_orig
        elif num_orig.empty:
            self.normalized_data_df = cat_norm
        else:
            self.normalized_data_df = pd.concat([num_orig, cat_norm], axis=1)

        print(f"Combined dataset: {self.normalized_data_df.shape[1]} features.")
        from IPython.display import display
        with pd.option_context("display.max_columns", None):
            display(self.normalized_data_df.head(5))
        return self.normalized_data_df

    # ------------------------------------------------------------------
    # STATISTICAL STATIONARITY TESTS
    # ------------------------------------------------------------------

    def test_constant_mean(self, columns: Optional[Sequence[str]] = None, chunks: int = 10):
        """
        Tests whether the multivariate mean stays stable across sequential data blocks
        using Wilks' Lambda (MANOVA approximation).
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        targets = [c for c in (columns or self.df.select_dtypes(include=["number"]).columns) if c != "count"]
        clean = self.df[targets].dropna().reset_index(drop=True)
        n, p = clean.shape

        if n < chunks * 2 or p == 0:
            print(f"Not enough data ({n} rows, {p} features) for {chunks} blocks.")
            return

        blocks = np.array_split(clean.values, chunks)
        grand_mean = clean.values.mean(axis=0)
        W = np.zeros((p, p))
        B = np.zeros((p, p))

        for block in blocks:
            if len(block) == 0:
                continue
            b_mean = block.mean(axis=0)
            diff_w = block - b_mean
            W += diff_w.T @ diff_w
            diff_b = b_mean - grand_mean
            B += len(block) * np.outer(diff_b, diff_b)

        W += np.eye(p) * 1e-4

        try:
            _, logdet_w = np.linalg.slogdet(W)
            _, logdet_t = np.linalg.slogdet(W + B)
            wilks = np.exp(logdet_w - logdet_t)
            df_deg = p * (chunks - 1)
            bartlett = -((n - 1) - (p + chunks) / 2.0) * (logdet_w - logdet_t)
            p_val = scipy.stats.chi2.sf(bartlett, df_deg)

            print("\n--- Mean Stationarity Test (MANOVA / Wilks' Lambda) ---")
            print(f"  Blocks          : {chunks}")
            print(f"  Features        : {p}  |  Rows: {n}")
            print(f"  Wilks' Lambda   : {wilks:.6f}")
            print(f"  Chi-Square stat : {bartlett:.4f}")
            print(f"  p-value         : {p_val:.6e}")
            if p_val < 0.05:
                print("  Result: Mean drift detected across blocks (p < 0.05).")
            else:
                print("  Result: No significant mean shift detected.")
        except np.linalg.LinAlgError:
            print("Matrix singularity error during Wilks' Lambda calculation.")

    def test_constant_covariance(self, columns: Optional[Sequence[str]] = None, chunks: int = 10):
        """
        Tests whether the covariance structure stays consistent across data blocks
        using Box's M approximation.
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        targets = [c for c in (columns or self.df.select_dtypes(include=["number"]).columns) if c != "count"]
        clean = self.df[targets].dropna().reset_index(drop=True)
        n, p = clean.shape

        if n < chunks * p:
            print(f"Not enough rows ({n}) for {p}×{p} block covariance matrices.")
            return

        blocks = np.array_split(clean.values, chunks)
        pooled = np.zeros((p, p))
        block_covs, block_sizes = [], []

        for block in blocks:
            nk = len(block)
            if nk <= 1:
                continue
            diff = block - block.mean(axis=0)
            cov = (diff.T @ diff) / (nk - 1) + np.eye(p) * 1e-4
            block_covs.append(cov)
            block_sizes.append(nk)
            pooled += (nk - 1) * cov

        N_g = sum(nk - 1 for nk in block_sizes)
        pooled /= N_g

        try:
            _, logdet_p = np.linalg.slogdet(pooled)
            M = 0.0
            for cov_k, nk in zip(block_covs, block_sizes):
                _, logdet_k = np.linalg.slogdet(cov_k)
                M += (nk - 1) * logdet_k
            box_m = N_g * logdet_p - M

            g = len(block_sizes)
            c1 = (((2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))) *
                  (sum(1.0 / (nk - 1) for nk in block_sizes) - 1.0 / N_g))
            df_m = (p * (p + 1) * (g - 1)) / 2.0
            chi2_stat = box_m * (1.0 - c1)
            p_val = scipy.stats.chi2.sf(chi2_stat, df_m)

            print("\n--- Covariance Stationarity Test (Box's M) ---")
            print(f"  Blocks       : {g}")
            print(f"  Box's M      : {box_m:.4f}")
            print(f"  Chi-Square   : {chi2_stat:.4f}  (df={df_m})")
            print(f"  p-value      : {p_val:.6e}")
            if p_val < 0.05:
                print("  Result: Covariance structure changes across blocks (p < 0.05).")
            else:
                print("  Result: Covariance structure appears stable.")
        except np.linalg.LinAlgError:
            print("Matrix singularity error during Box's M calculation.")

    def test_row_independence(self, columns: Optional[Sequence[str]] = None, max_lag: Optional[int] = None):
        """
        Tests for serial autocorrelation using the multivariate Ljung-Box portmanteau test.
        """
        if self.df is None:
            print("No data loaded yet.")
            return

        targets = [c for c in (columns or self.df.select_dtypes(include=["number"]).columns) if c != "count"]
        clean = self.df[targets].dropna().reset_index(drop=True)
        n, p = clean.shape

        if n < 5:
            print("Not enough rows for lag calculations.")
            return

        if max_lag is None:
            max_lag = max(1, min(int(np.floor(np.log(n))), n - 2))

        data = clean.values - clean.values.mean(axis=0)
        c0 = (data.T @ data) / n
        c0_inv = np.linalg.pinv(c0)

        q = 0.0
        for lag in range(1, max_lag + 1):
            c_lag = (data[lag:].T @ data[:-lag]) / n
            q += np.trace(c_lag.T @ c0_inv @ c_lag @ c0_inv) / (n - lag)
        q *= n * (n + 2)

        df_test = (p ** 2) * max_lag
        p_val = scipy.stats.chi2.sf(q, df_test)

        print("\n--- Row Independence Test (Multivariate Ljung-Box) ---")
        print(f"  Max lag      : {max_lag}")
        print(f"  Q statistic  : {q:.4f}  (df={df_test})")
        print(f"  p-value      : {p_val:.6e}")
        if p_val < 0.05:
            print("  Result: Serial autocorrelation detected (p < 0.05).")
        else:
            print("  Result: Rows appear to be independent observations.")

    # ------------------------------------------------------------------
    # DISTRIBUTION MODELLING
    # ------------------------------------------------------------------

    def estimate_joint_normal(self, columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        Fits a multivariate normal distribution to the numeric data (MLE with Bessel's correction).
        Returns the mean vector, covariance matrix, and Mahalanobis distances.
        """
        if self.df is None:
            print("No data loaded yet.")
            return {}

        targets = [c for c in (columns or self.df.select_dtypes(include=["number"]).columns) if c != "count"]
        clean = self.df[targets].dropna().reset_index(drop=True)
        n, p = clean.shape

        if n <= p:
            print(f"Row count ({n}) must exceed feature count ({p}) for multivariate normal fitting.")
            return {}

        X = clean.values
        mu = X.mean(axis=0)
        diff = X - mu
        S = (diff.T @ diff) / (n - 1)
        S_stable = S + np.eye(p) * 1e-4

        try:
            from scipy.stats import multivariate_normal
            dist = multivariate_normal(mean=mu, cov=S_stable, allow_singular=True)
            S_inv = np.linalg.pinv(S_stable)
            md = np.array([d @ S_inv @ d for d in diff])
            t95 = scipy.stats.chi2.ppf(0.95, df=p)
            t99 = scipy.stats.chi2.ppf(0.99, df=p)
            anomalies = int(np.sum(md > t95))

            print("\n--- Multivariate Normal Distribution Fit ---")
            print(f"  Features  : {p}  |  Rows: {n}")
            print(f"  95% threshold : {t95:.4f}  |  99%: {t99:.4f}")
            print(f"  Outliers (>95%): {anomalies} rows ({anomalies / n * 100:.1f}%)")

            return {
                "mean_vector": mu,
                "covariance_matrix": S,
                "distribution_object": dist,
                "mahalanobis_distances": md,
                "threshold_95": t95,
                "threshold_99": t99,
            }
        except Exception as e:
            print(f"Distribution fitting error: {e}")
            return {}

    def instantiate_macro_clt_distribution(self, columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        Builds the CLT sampling distribution of the empirical mean vector:
        mu_hat_n ~ N(mu_hat, (1/n) * S).
        """
        mle = self.estimate_joint_normal(columns=columns)
        if not mle:
            return {}

        from scipy.stats import multivariate_normal
        mu = mle["mean_vector"]
        S = mle["covariance_matrix"]
        p = len(mu)
        cols = columns or [c for c in self.df.select_dtypes(include=["number"]).columns if c != "count"]
        n = len(self.df[cols].dropna())

        clt_cov = S / n + np.eye(p) * 1e-5

        try:
            dist = multivariate_normal(mean=mu, cov=clt_cov, allow_singular=True)
            f_crit = scipy.stats.f.ppf(0.95, dfn=p, dfd=n - p)
            ellipsoid = (p * (n - 1) / (n - p)) * f_crit

            print("\n--- CLT Sampling Distribution ---")
            print(f"  n={n}  |  Max SE: {np.sqrt(np.diag(clt_cov)).max():.6f}")
            print(f"  Confidence ellipsoid scale: {ellipsoid:.4f}")

            return {
                "clt_mean_vector": mu,
                "clt_covariance_matrix": clt_cov,
                "clt_distribution_object": dist,
                "ellipsoid_scale_factor": ellipsoid,
            }
        except Exception as e:
            print(f"CLT distribution error: {e}")
            return {}

    def compute_empirical_pca(self, columns: Optional[Sequence[str]] = None, show_plot: bool = True) -> Dict[str, Any]:
        """
        Runs PCA on numeric data and shows a 2×3 dashboard of key outputs
        (scree plot, cumulative variance, component loadings, biplot, etc.).
        """
        if self.df is None:
            print("No data loaded yet.")
            return {}

        targets = [c for c in (columns or self.df.select_dtypes(include=["number"]).columns) if c != "count"]
        clean = self.df[targets].dropna().reset_index(drop=True)
        n, p = clean.shape

        if n <= p:
            print(f"Need more rows ({n}) than features ({p}) for PCA.")
            return {}

        X = clean.values
        mu = X.mean(axis=0)
        X_c = X - mu
        S = (X_c.T @ X_c) / (n - 1) + np.eye(p) * 1e-8

        eigenvalues, eigenvectors = np.linalg.eigh(S)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        explained = eigenvalues / eigenvalues.sum()
        cumulative = np.cumsum(explained)
        scores = X_c @ eigenvectors

        result = {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "explained_variance_ratio": explained,
            "cumulative_variance": cumulative,
            "pca_scores": scores,
            "feature_names": targets,
        }

        if show_plot:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    "Scree Plot", "Cumulative Variance", "PC1 Loadings",
                    "PC2 Loadings", "PC1 vs PC2 Scores", "Correlation Circle",
                ),
            )
            comp_idx = list(range(1, min(p + 1, 11)))

            fig.add_trace(go.Bar(x=comp_idx, y=explained[:10], name="Explained",
                                 marker_color="steelblue"), row=1, col=1)
            fig.add_trace(go.Scatter(x=comp_idx, y=cumulative[:10], mode="lines+markers",
                                     name="Cumulative", line_color="crimson"), row=1, col=2)
            fig.add_trace(go.Bar(x=targets, y=eigenvectors[:, 0], name="PC1",
                                 marker_color="mediumseagreen"), row=1, col=3)
            if p >= 2:
                fig.add_trace(go.Bar(x=targets, y=eigenvectors[:, 1], name="PC2",
                                     marker_color="darkorchid"), row=2, col=1)
                fig.add_trace(go.Scatter(x=scores[:, 0], y=scores[:, 1], mode="markers",
                                         marker=dict(size=5, opacity=0.6, color="royalblue"),
                                         name="Scores"), row=2, col=2)
                theta = np.linspace(0, 2 * np.pi, 100)
                fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines",
                                         line=dict(color="grey", dash="dash"), showlegend=False), row=2, col=3)
                for feat, lx, ly in zip(targets, eigenvectors[:, 0], eigenvectors[:, 1]):
                    fig.add_trace(go.Scatter(x=[0, lx], y=[0, ly], mode="lines+text",
                                             text=["", feat], textposition="top center",
                                             line=dict(color="tomato"), showlegend=False), row=2, col=3)

            fig.update_layout(height=700, width=1200, title_text="<b>PCA Dashboard</b>",
                               showlegend=False, template="plotly_white")
            fig.show()

        return result


# ---------------------------------------------------------------------------
# PlottingMethods
# ---------------------------------------------------------------------------

class PlottingMethods:
    """
    Standalone chart generators that work with any DataFrame or JSON payload.
    Each method returns a dict with status, metadata, and an HTML figure string.
    Render the output using display_image().
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df

    def get_methods_info(self) -> Dict[str, Any]:
        """Returns the name, signature, and docstring of every public method."""
        info = []
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if name.startswith("_") or name == "get_methods_info":
                continue
            info.append({
                "method": name,
                "signature": str(inspect.signature(method)),
                "description": (method.__doc__ or "No description.").strip(),
            })
        return {"status": "success", "response": info}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_data(self, data: Any) -> pd.DataFrame:
        """
        Accepts a DataFrame, list of records, dict, or JSON string and
        always returns a DataFrame. Raises ValueError on bad input.
        """
        if data is None or (isinstance(data, str) and data == '{"records":[]}'):
            if self.df is not None:
                return self.df.copy()
            raise ValueError("No data provided and no default DataFrame is set.")

        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("The DataFrame is empty.")
            return data.copy()

        if isinstance(data, list):
            if not data:
                raise ValueError("The list of records is empty.")
            return pd.DataFrame(data)

        if isinstance(data, dict):
            records = data.get("records", data)
            if isinstance(records, dict):
                records = [records]
            if isinstance(records, list):
                if not records:
                    raise ValueError("The records list is empty.")
                return pd.DataFrame(records)
            return pd.DataFrame([data])

        if isinstance(data, str):
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSON: {exc}") from exc

            if isinstance(parsed, list):
                return pd.DataFrame(parsed)
            if isinstance(parsed, dict):
                records = parsed.get("records", parsed)
                if isinstance(records, list):
                    return pd.DataFrame(records)
                return pd.DataFrame([parsed])

        raise ValueError(f"Unsupported data type: {type(data)}")

    @staticmethod
    def _ok(fig, label: str) -> Dict[str, Any]:
        html = pio.to_html(fig, full_html=False,
                           config={"displaylogo": False, "responsive": True},
                           include_plotlyjs=True)
        fig_id = f"fig_{uuid.uuid4().hex[:8]}"
        html = html.replace("<div>", f'<div id="{fig_id}">', 1)
        return {
            "status": "success",
            "response": {
                "meta_data": {},
                "data": json.dumps({"figure": html}),
                "message": json.dumps({"message": label}),
            },
        }

    @staticmethod
    def _err(msg: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "response": {
                "meta_data": {},
                "data": json.dumps({"figure": ""}),
                "message": json.dumps({"message": str(msg)}),
            },
        }

    # ------------------------------------------------------------------
    # Chart methods
    # ------------------------------------------------------------------

    def plot_bar_chart(
        self,
        x: str = "date",
        y: str = "value",
        color: Optional[str] = None,
        text: Optional[str] = None,
        title: str = "",
        barmode: str = "stack",
        hover_data: Optional[Union[str, List[str]]] = None,
        data: Any = '{"records":[]}',
        meta_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Draws a bar chart. Supports grouped or stacked bars, hover data, and color grouping."""
        try:
            df = self._parse_data(data)
            if x not in df.columns or y not in df.columns:
                raise KeyError(f"Columns '{x}' and '{y}' must exist in the dataset.")

            df[y] = pd.to_numeric(df[y], errors="coerce")

            if isinstance(hover_data, str):
                try:
                    hover_data = json.loads(hover_data)
                except json.JSONDecodeError:
                    hover_data = hover_data.split(",") if "," in hover_data else None
            if hover_data:
                hover_data = [c for c in hover_data if c in df.columns]

            cat_orders: Dict[str, list] = {}
            if color and color in df.columns:
                df.dropna(subset=[color], inplace=True)
                labels = sorted(df[color].unique()) if not any(k in color.lower() for k in ("month", "week")) else df[color].unique()
                df[color] = pd.Categorical(df[color], categories=labels, ordered=True)
                cat_orders[color] = list(labels)

            x_labels = df[x].unique()
            df[x] = pd.Categorical(df[x], categories=x_labels, ordered=True)
            cat_orders[x] = list(x_labels)

            fig = px.bar(df, x=x, y=y, color=color, title=title or None,
                         text=text, hover_data=hover_data or None, category_orders=cat_orders)
            fig.update_layout(xaxis_title=x, yaxis_title=y,
                               uniformtext_minsize=8, uniformtext_mode="hide", barmode=barmode)
            return self._ok(fig, "Bar chart plotted")
        except Exception as exc:
            return self._err(exc)

    def plot_pie_chart(
        self,
        names: str = "category",
        values: str = "value",
        title: str = "",
        hole: Optional[float] = None,
        data: Any = '{"records":[]}',
        meta_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Draws a pie or donut chart. Set hole (0–1) for a donut style."""
        try:
            df = self._parse_data(data)
            if names not in df.columns or values not in df.columns:
                raise KeyError(f"Columns '{names}' and '{values}' must exist in the dataset.")
            df[values] = pd.to_numeric(df[values], errors="coerce")
            fig = px.pie(df, names=names, values=values, title=title or None, hole=hole)
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            return self._ok(fig, "Pie chart plotted")
        except Exception as exc:
            return self._err(exc)

    def plot_histogram(
        self,
        x: str = "value",
        bins: Optional[List[float]] = None,
        title: str = "",
        color: Optional[str] = None,
        data: Any = '{"records":[]}',
        meta_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Draws a histogram. Optionally pass custom bin edges as a list."""
        try:
            df = self._parse_data(data)
            if x not in df.columns:
                raise KeyError(f"Column '{x}' not found in the dataset.")
            df[x] = pd.to_numeric(df[x], errors="coerce")
            kwargs: Dict[str, Any] = dict(x=x, title=title or None)
            if color and color in df.columns:
                kwargs["color"] = color
            fig = px.histogram(df, **kwargs)
            if bins:
                fig.update_traces(xbins=dict(start=bins[0], end=bins[-1],
                                             size=(bins[-1] - bins[0]) / (len(bins) - 1)))
            fig.update_layout(template="plotly_white")
            return self._ok(fig, "Histogram plotted")
        except Exception as exc:
            return self._err(exc)

    def plot_heat_map(
        self,
        values: str = "value",
        index: str = "row",
        columns: str = "col",
        aggregate_method: str = "mean",
        fill_value: Union[int, float] = 0,
        title: str = "Heatmap",
        width: Optional[int] = None,
        data: Any = '{"records":[]}',
        meta_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Pivots long-form data and renders a heatmap.
        The aggregate_method parameter accepts any pandas aggfunc ('mean', 'sum', etc.).
        """
        agg = kwargs.get("aggregade_method", aggregate_method)
        try:
            df = self._parse_data(data).copy()
            missing = [c for c in [values, index, columns] if c not in df.columns]
            if missing:
                raise KeyError(f"Columns not found: {missing}")
            df[values] = pd.to_numeric(df[values], errors="coerce")
            pivot = df.pivot_table(values=values, index=index, columns=columns,
                                   aggfunc=agg, fill_value=fill_value)
            fig = px.imshow(
                pivot,
                labels=dict(x=columns, y=index, color=values),
                x=pivot.columns.astype(str).tolist(),
                y=pivot.index.astype(str).tolist(),
                title=title or None,
                text_auto=True,
            )
            layout = dict(template="plotly_white",
                          xaxis=dict(type="category"),
                          yaxis=dict(type="category"))
            if width:
                layout["width"] = width
            fig.update_layout(**layout)
            return self._ok(fig, "Heatmap plotted")
        except Exception as exc:
            return self._err(exc)

    def plot_sankey_diagram(
        self,
        source_column: str = "source",
        target_column: str = "target",
        values: str = "value",
        title: str = "Sankey Diagram",
        data: Any = '{"records":[]}',
        meta_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Draws a Sankey flow diagram mapping sources to targets, weighted by a value column."""
        try:
            df = self._parse_data(data).copy()
            if not {source_column, target_column, values}.issubset(df.columns):
                raise KeyError("source_column, target_column, and values must all exist in the dataset.")
            df[values] = pd.to_numeric(df[values], errors="coerce")
            df.dropna(subset=[source_column, target_column, values], inplace=True)

            all_nodes = sorted(
                set(df[source_column].astype(str)).union(set(df[target_column].astype(str)))
            )
            idx = {node: i for i, node in enumerate(all_nodes)}
            sources = df[source_column].astype(str).map(idx).tolist()
            targets = df[target_column].astype(str).map(idx).tolist()
            vals = df[values].tolist()

            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20,
                          line=dict(color="black", width=0.5),
                          label=all_nodes, color="rgba(31,119,180,0.8)"),
                link=dict(source=sources, target=targets, value=vals),
            )])
            fig.update_layout(title_text=title or None, font_size=10, template="plotly_white")
            return self._ok(fig, "Sankey diagram plotted")
        except Exception as exc:
            return self._err(exc)

    def plot_sunburst_from_hierarchy(
        self,
        path: Sequence[str] = ("parent", "name"),
        values: str = "value",
        title: str = "Sunburst",
        data: Any = '{"records":[]}',
        meta_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds a sunburst chart from a two-column parent/child adjacency list.
        Cycle detection is included to prevent infinite loops.
        """
        def build_hierarchy(df_src: pd.DataFrame, path_cols: list, val_col: str, root: str = "Root") -> pd.DataFrame:
            df_h = df_src.copy()
            p_src, p_tgt = path_cols[0], path_cols[1]
            df_h[p_src] = df_h[p_src].replace([None, "", pd.NA], root).astype(str).str.strip()
            df_h[p_tgt] = df_h[p_tgt].astype(str).str.strip()
            df_h[val_col] = pd.to_numeric(df_h[val_col], errors="coerce").fillna(0)

            adjacency = dict(zip(df_h[p_tgt], df_h[p_src]))
            for child, parent in adjacency.items():
                visited = {child}
                curr = parent
                while curr in adjacency:
                    if curr in visited:
                        raise ValueError(f"Cycle detected at node: {curr}")
                    visited.add(curr)
                    curr = adjacency[curr]

            values_map = {r[p_tgt]: r[val_col] for _, r in df_h.iterrows()}
            rows = []
            for _, row in df_h.iterrows():
                chain = [row[p_tgt]]
                curr = row[p_tgt]
                while curr in adjacency and adjacency[curr] != root:
                    curr = adjacency[curr]
                    chain.insert(0, curr)
                chain.insert(0, root)
                row_dict = {f"level_{i}": chain[i] for i in range(len(chain))}
                row_dict[val_col] = values_map.get(row[p_tgt], 0)
                rows.append(row_dict)
            return pd.DataFrame(rows)

        try:
            df = self._parse_data(data)
            h_df = build_hierarchy(df, list(path), values)
            path_cols = [c for c in h_df.columns if c != values]
            fig = px.sunburst(h_df, path=path_cols, values=values, title=title or None)
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide", template="plotly_white")
            return self._ok(fig, "Sunburst chart plotted")
        except Exception as exc:
            return self._err(exc)

    def plot_multi_column_bar_graph(
        self,
        xLabel: str = "category",
        value_vars: Optional[Sequence[str]] = None,
        title: str = "Multi-Column Bar Chart",
        orientation: str = "v",
        hover_data: Optional[Sequence[str]] = None,
        barmode: str = "group",
        data: Any = '{"records":[]}',
        meta_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Melts wide-form data into long-form and draws a multi-variable bar chart.
        Useful for comparing several metrics side by side.
        """
        value_vars = value_vars or []
        hover_cols = list(hover_data) if isinstance(hover_data, (list, tuple, set)) else []

        try:
            df = self._parse_data(data)
            if df.empty:
                return self._err("No data records found.")
            if xLabel not in df.columns:
                raise KeyError(f"x-axis column '{xLabel}' not found.")

            valid_vars = [v for v in value_vars if v in df.columns]
            if not valid_vars:
                raise KeyError("None of the requested value columns exist in the dataset.")

            needed = list(set([xLabel] + hover_cols + valid_vars))
            work = df[[c for c in needed if c in df.columns]].copy()
            melted = work.melt(
                id_vars=[xLabel] + [h for h in hover_cols if h in work.columns],
                value_vars=valid_vars,
                var_name="Group", value_name="Value",
            )
            melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce").fillna(0)

            is_v = orientation.lower() == "v"
            fig = px.bar(
                melted,
                x=xLabel if is_v else "Value",
                y="Value" if is_v else xLabel,
                color="Group", barmode=barmode,
                title=title or None,
                hover_data=[h for h in hover_cols if h in melted.columns] or None,
                orientation=orientation.lower() if orientation.lower() in ("v", "h") else "v",
            )
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(font=dict(size=10), orientation="h", x=0.5, xanchor="center", y=1.02),
            )
            return self._ok(fig, "Multi-column bar chart plotted")
        except Exception as exc:
            return self._err(exc)

    def plot_flow_chart_plotly(
        self,
        data: Any = '{"records":[]}',
        meta_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        data_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Draws an interactive flowchart using NetworkX for layout and Plotly for rendering.
        Input data must be a dict (or JSON string) with 'nodes' and 'edges' keys.

        Node dict keys: label, fillcolor, fontcolor
        Edge dict keys: start, end, label, color, penwidth
        """
        import networkx as nx

        try:
            if isinstance(data, str):
                try:
                    parsed = json.loads(data)
                    records = parsed.get("records", parsed)
                except Exception:
                    records = {}
            elif isinstance(data, dict):
                records = data.get("records", data)
            elif isinstance(data, pd.DataFrame):
                records = {"nodes": data.to_dict(orient="records"), "edges": []}
            else:
                records = {}

            if not isinstance(records, dict):
                records = {}

            edges_raw = records.get("edges", []) or []
            nodes_raw = records.get("nodes", []) or []

            if not edges_raw and not nodes_raw:
                return self._err("No nodes or edges found in input data.")

            G = nx.DiGraph()
            edge_labels: Dict[Tuple, str] = {}
            edge_styles: Dict[Tuple, Dict] = {}

            for e in edges_raw:
                if not isinstance(e, dict):
                    continue
                start = str(e.get("start", "")).split(":")[0].strip()
                end = str(e.get("end", "")).split(":")[0].strip()
                if not start or not end:
                    continue
                G.add_edge(start, end)
                edge_labels[(start, end)] = e.get("label", "")
                edge_styles[(start, end)] = {
                    "color": e.get("color", "#333333"),
                    "width": float(e.get("penwidth", 2)),
                }

            for np_info in nodes_raw:
                if not isinstance(np_info, dict):
                    continue
                lbl = str(np_info.get("label", "")).strip()
                if lbl:
                    if lbl not in G.nodes:
                        G.add_node(lbl)
                    G.nodes[lbl].update(np_info)

            if len(G.nodes) == 0:
                raise ValueError("No valid nodes found.")

            roots = [n for n, d in G.in_degree() if d == 0]
            for node in G.nodes:
                depth = 0
                for root in roots:
                    if nx.has_path(G, root, node):
                        depth = max(depth, len(nx.shortest_path(G, root, node)) - 1)
                G.nodes[node]["layer"] = depth

            pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")

            edge_traces = []
            arrow_x: List[float] = []
            arrow_y: List[float] = []

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                style = edge_styles.get(edge, {"color": "#333333", "width": 2})
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    line=dict(width=style["width"], color=style["color"]),
                    hoverinfo="none", mode="lines", showlegend=False,
                ))
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = x1 - x0, y1 - y0
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    ux, uy = dx / length, dy / length
                    arrow_x += [mx, mx - 0.04 * ux + 0.02 * uy, mx - 0.04 * ux - 0.02 * uy, mx, None]
                    arrow_y += [my, my - 0.04 * uy - 0.02 * ux, my - 0.04 * uy + 0.02 * ux, my, None]

            if arrow_x:
                edge_traces.append(go.Scatter(
                    x=arrow_x, y=arrow_y, fill="toself",
                    fillcolor="#333333", line=dict(color="#333333", width=1),
                    hoverinfo="none", mode="lines", showlegend=False,
                ))

            node_x, node_y, node_text, node_color, node_font_color = [], [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                info = G.nodes[node]
                node_text.append(f"<b>{info.get('label', node)}</b>")
                node_color.append(info.get("fillcolor", "#E5ECF6"))
                node_font_color.append(info.get("fontcolor", "#000000"))

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(size=45, color=node_color,
                            line=dict(width=2, color="#1A1A1A"), shape="square"),
                text=node_text, textposition="middle center",
                textfont=dict(color=node_font_color, size=11),
                hoverinfo="text", showlegend=False,
            )

            el_x, el_y, el_text = [], [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                lbl = edge_labels.get(edge, "")
                if lbl:
                    el_x.append(x0 + (x1 - x0) * 0.35)
                    el_y.append(y0 + (y1 - y0) * 0.35)
                    el_text.append(f"<span style='background:white;padding:2px'>{lbl}</span>")

            label_trace = go.Scatter(
                x=el_x, y=el_y, mode="text", text=el_text,
                textposition="top center", textfont=dict(size=9, color="#555"),
                hoverinfo="none", showlegend=False,
            )

            fig = go.Figure(
                data=[*edge_traces, node_trace, label_trace],
                layout=go.Layout(
                    hovermode="closest",
                    margin=dict(b=40, l=40, r=40, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                               scaleanchor="x", scaleratio=1),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                ),
            )
            return self._ok(fig, "Flowchart plotted")
        except Exception as exc:
            return self._err(exc)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def display_image(self, result: Dict[str, Any]) -> None:
        """
        Renders the figure HTML returned by any plot method directly in a notebook cell.
        """
        from IPython.display import HTML, display

        if not isinstance(result, dict):
            print("Invalid result format.")
            return

        if result.get("status") != "success":
            raw = result.get("response", {}).get("message") or "Unknown error"
            try:
                msg = json.loads(raw) if isinstance(raw, str) else raw
                print(f"Plot failed: {msg.get('message', raw) if isinstance(msg, dict) else msg}")
            except Exception:
                print(f"Plot failed: {raw}")
            return

        try:
            payload = result["response"].get("data", "{}")
            if isinstance(payload, str):
                payload = json.loads(payload)
            html = payload.get("figure", "")
            if html:
                display(HTML(html))
            else:
                print("No figure data found in result.")
        except Exception as exc:
            print(f"Could not render figure: {exc}")
