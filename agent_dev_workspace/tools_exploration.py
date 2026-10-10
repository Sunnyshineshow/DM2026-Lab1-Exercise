import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index_1: int, doc_index_2: int) -> Dict[str, Any]:
        """Calculates cosine similarity between two documents with term breakdown.

        Args:
            doc_index_1: 0-based row index of the first document.
            doc_index_2: 0-based row index of the second document.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found. Call build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]

        if doc_index_1 < 0 or doc_index_1 >= n_docs:
            return {"error": f"doc_index_1 ({doc_index_1}) out of bounds for {n_docs} documents."}

        if doc_index_2 < 0 or doc_index_2 >= n_docs:
            return {"error": f"doc_index_2 ({doc_index_2}) out of bounds for {n_docs} documents."}

        # Convert ONLY the two requested rows to dense 1D arrays
        u = session.feature_matrix[doc_index_1].toarray().ravel()
        v = session.feature_matrix[doc_index_2].toarray().ravel()

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u == 0 or norm_v == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(u, v) / (norm_u * norm_v))
            if np.isnan(similarity):
                similarity = 0.0

        # Term breakdown for terms occurring in at least one of the two documents
        term_breakdown = []
        for name, count1, count2 in zip(session.feature_names, u, v):
            if count1 > 0 or count2 > 0:
                term_breakdown.append({
                    "term": name,
                    "doc_1_count": int(count1),
                    "doc_2_count": int(count2),
                    "product": int(count1 * count2)
                })

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity": round(similarity, 4),
            "term_breakdown": term_breakdown
        }

        session.store_result("cosine_similarity_tool", {
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2
        }, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Heatmap and Pearson correlation matrix over the top-20 terms sorted by variance descending."""
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found. Call build_dtm_tool first."}

        X_sparse = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Compute column variances efficiently using sparse operations: Var(X) = E[X^2] - (E[X])^2
        mean_X = np.asarray(X_sparse.mean(axis=0)).ravel()
        mean_X2 = np.asarray(X_sparse.power(2).mean(axis=0)).ravel()
        variances = mean_X2 - (mean_X ** 2)

        # Sort terms by descending variance
        sorted_indices = np.argsort(-variances)
        top_indices = sorted_indices[:20]

        top_terms = feature_names[top_indices].tolist()
        
        # Convert ONLY the top-20 selected feature columns to dense matrix
        X_top = X_sparse[:, top_indices].toarray()

        n_features = len(top_terms)
        corr_matrix = np.zeros((n_features, n_features))

        stds = np.std(X_top, axis=0)

        for i in range(n_features):
            for j in range(n_features):
                # Constant feature (std == 0) yields undefined Pearson correlation -> set to 0.0 (including diagonal)
                if stds[i] == 0 or stds[j] == 0:
                    corr_matrix[i, j] = 0.0
                elif i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    r = np.corrcoef(X_top[:, i], X_top[:, j])[0, 1]
                    corr_matrix[i, j] = 0.0 if np.isnan(r) else float(r)

        # Build heatmap figure
        fig, ax = plt.subplots(figsize=(max(6, n_features * 0.8), max(5, n_features * 0.7)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax
        )
        ax.set_title(f"Feature Correlation Matrix (Top-{n_features} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        # Round matrix entries for clean summary
        corr_matrix_rounded = [[round(val, 4) for val in row] for row in corr_matrix.tolist()]

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "n_terms": n_features,
            "feature_names": top_terms,
            "correlation_matrix": corr_matrix_rounded
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
