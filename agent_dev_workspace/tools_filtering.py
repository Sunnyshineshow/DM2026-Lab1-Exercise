import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import spearmanr
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Flags near-constant terms by variance for removal based on a variance threshold.

        Args:
            threshold: Minimum variance threshold to keep a term.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found. Call build_dtm_tool first."}

        X = session.feature_matrix.toarray()
        variances = np.var(X, axis=0, ddof=0)

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_mask = variances > threshold
        kept_features = [name for name, keep in zip(session.feature_names, kept_mask) if keep]
        removed_features = [name for name, keep in zip(session.feature_names, kept_mask) if not keep]

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "n_kept": len(kept_features),
            "n_removed": len(removed_features),
            "kept_features": kept_features,
            "removed_features": removed_features,
            "variances": {term: float(var) for term, var in zip(session.feature_names, variances)}
        }

        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=df_report)
        return summary

    @tool
    def pearson_filter_tool(target_class: str) -> Dict[str, Any]:
        """Calculates Pearson correlation between each term count and a target class label (one-vs-rest).

        Args:
            target_class: The class label to compute correlation against.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"error": "Missing feature matrix or category labels. Run build_dtm_tool first."}

        labels = np.array(session.labels)
        X = session.feature_matrix.toarray()

        if len(labels) != X.shape[0]:
            return {"error": f"Label count ({len(labels)}) does not match DTM row count ({X.shape[0]})."}

        if target_class not in labels:
            available_classes = sorted(list(set(labels)))
            return {"error": f"Target class '{target_class}' not found in labels. Available classes: {available_classes}."}

        y = (labels == target_class).astype(float)

        correlations = []
        y_std = np.std(y)

        for col_idx in range(X.shape[1]):
            x_col = X[:, col_idx]
            x_std = np.std(x_col)
            if x_std == 0 or y_std == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(x_col, y)[0, 1])
                if np.isnan(r):
                    r = 0.0
            correlations.append(r)

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": correlations
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "correlations": {term: round(corr, 4) for term, corr in zip(session.feature_names, correlations)}
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class}, summary, full_report=df_report)
        return summary

    @tool
    def spearman_filter_tool(target_class: str) -> Dict[str, Any]:
        """Calculates Spearman rank correlation between each term count and a target class label.

        Args:
            target_class: The class label to compute rank correlation against.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"error": "Missing feature matrix or category labels. Run build_dtm_tool first."}

        labels = np.array(session.labels)
        X = session.feature_matrix.toarray()

        if len(labels) != X.shape[0]:
            return {"error": f"Label count ({len(labels)}) does not match DTM row count ({X.shape[0]})."}

        if target_class not in labels:
            available_classes = sorted(list(set(labels)))
            return {"error": f"Target class '{target_class}' not found in labels. Available classes: {available_classes}."}

        y = (labels == target_class).astype(float)

        correlations = []
        for col_idx in range(X.shape[1]):
            x_col = X[:, col_idx]
            if np.all(x_col == x_col[0]) or np.all(y == y[0]):
                r = 0.0
            else:
                res = spearmanr(x_col, y)
                r = float(res.statistic) if hasattr(res, 'statistic') else float(res[0])
                if np.isnan(r):
                    r = 0.0
            correlations.append(r)

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "spearman_r": correlations
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "correlations": {term: round(corr, 4) for term, corr in zip(session.feature_names, correlations)}
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class}, summary, full_report=df_report)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
