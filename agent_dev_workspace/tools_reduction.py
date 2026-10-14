import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Literal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> Dict[str, Any]:
        """Reduces feature matrix dimensions (2D scatter plot required) via PCA, t-SNE, or UMAP.

        Args:
            method: 'pca', 'tsne', or 'umap'.
            n_components: Number of dimensions to reduce to (must be 2 for scatter plotting).
            random_state: Seed for reproducible results.
            perplexity: t-SNE perplexity (must be > 0 and < number of samples).
            n_neighbors: UMAP n_neighbors (must be >= 2 and < number of samples).
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix found. Call build_dtm_tool first."}

        if n_components != 2:
            return {"error": f"n_components must be 2 for 2D scatter plotting, got {n_components}."}

        if method not in ["pca", "tsne", "umap"]:
            return {"error": f"Invalid method '{method}'. Must be 'pca', 'tsne', or 'umap'."}

        X = session.feature_matrix.toarray()
        n_samples, n_features = X.shape

        if n_samples < 2 or n_features < 2:
            return {"error": f"Insufficient data for 2D reduction: feature matrix shape is ({n_samples}, {n_features}). At least 2 samples and 2 features required."}

        # Validate label sources rigorously
        df_labels = session.dataframe["category_name"].tolist() if (session.dataframe is not None and "category_name" in session.dataframe.columns) else None

        if session.labels is not None and df_labels is not None:
            sess_labels = list(session.labels)
            if len(sess_labels) != n_samples:
                return {"error": f"session.labels count ({len(sess_labels)}) does not match DTM row count ({n_samples})."}
            if len(df_labels) != n_samples:
                return {"error": f"DataFrame row count ({len(df_labels)}) does not match DTM row count ({n_samples})."}
            if sess_labels != df_labels:
                return {"error": "session.labels values do not match DataFrame 'category_name' values."}
            labels = sess_labels
        elif session.labels is not None:
            sess_labels = list(session.labels)
            if len(sess_labels) != n_samples:
                return {"error": f"session.labels count ({len(sess_labels)}) does not match DTM row count ({n_samples})."}
            labels = sess_labels
        elif df_labels is not None:
            if len(df_labels) != n_samples:
                return {"error": f"DataFrame row count ({len(df_labels)}) does not match DTM row count ({n_samples})."}
            labels = df_labels
        else:
            return {"error": "No category labels found in session. Run load_dataset_tool or build_dtm_tool first."}

        # Check for missing/empty label values
        if any(pd.isna(lbl) or str(lbl).strip() == "" for lbl in labels):
            return {"error": "Category labels contain missing or empty values."}

        # Validate parameters early
        if method == "tsne":
            if perplexity <= 0 or perplexity >= n_samples:
                return {"error": f"t-SNE perplexity ({perplexity}) must be > 0 and < number of samples ({n_samples})."}

        if method == "umap":
            if n_neighbors < 2 or n_neighbors >= n_samples:
                return {"error": f"UMAP n_neighbors ({n_neighbors}) must be >= 2 and less than number of samples ({n_samples})."}

        # Perform dimensionality reduction
        explained_variance_ratio = None
        if method == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = [float(v) for v in reducer.explained_variance_ratio_]
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
            coords = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(X)

        coords_list = coords.tolist()

        # Build required 2D scatter plot
        fig, ax = plt.subplots(figsize=(7, 5))
        df_plot = pd.DataFrame({
            "Component 1": coords[:, 0],
            "Component 2": coords[:, 1],
            "Category": labels
        })
        sns.scatterplot(
            data=df_plot,
            x="Component 1",
            y="Component 2",
            hue="Category",
            style="Category",
            s=100,
            ax=ax
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        plt.tight_layout()

        session.pending_figure = fig
        session.artifacts[f"reduced_{method}"] = coords_list

        result_id = session.next_result_id("reduction")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": 2,
            "n_samples": n_samples,
            "reduced_coordinates": [[round(x, 4), round(y, 4)] for x, y in coords_list]
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = [round(v, 4) for v in explained_variance_ratio]

        session.store_result("reduce_dimensions_tool", {"method": method}, summary)
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes document category labels into a binary indicator matrix."""
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "No working DataFrame or 'category_name' column found. Load a dataset first."}

        df_labels = session.dataframe["category_name"].tolist()

        # Check for missing/null label values in DataFrame
        if any(pd.isna(lbl) or str(lbl).strip() == "" for lbl in df_labels):
            return {"error": "DataFrame 'category_name' contains missing or empty label values."}

        labels = df_labels

        # Preserve alphabetical order for categories
        categories = sorted(list(set(labels)))
        binary_cols = [f"category_{cat}" for cat in categories]

        # Generate binary matrix preserving original document row order
        binary_matrix = []
        for label in labels:
            row = [1 if label == cat else 0 for cat in categories]
            binary_matrix.append(row)

        df_onehot = pd.DataFrame(binary_matrix, columns=binary_cols)
        session.artifacts["binarized_labels"] = df_onehot

        result_id = session.next_result_id("binarize")
        summary = {
            "result_id": result_id,
            "n_documents": len(labels),
            "n_categories": len(categories),
            "categories": categories,
            "binary_columns": binary_cols,
            "shape": [len(labels), len(categories)],
            "binary_matrix": binary_matrix
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
