import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def build_dtm_tool(
        text_column: str = "text",
        ngram_range: List[int] = [1, 1],
        min_df: int = 1,
        max_df: float = 1.0,
        lowercase: bool = True
    ) -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from the dataset text using CountVectorizer.

        Args:
            text_column: The column in session.dataframe containing document texts.
            ngram_range: The lower and upper boundary of range of n-values for n-grams, e.g. [1, 1].
            min_df: Ignore terms that have a document frequency strictly lower than this threshold.
            max_df: Ignore terms that have a document frequency strictly higher than this threshold.
            lowercase: Whether to convert all characters to lowercase before tokenizing.
        """
        if session.dataframe is None or text_column not in session.dataframe.columns:
            return {"error": f"DataFrame is empty or missing column '{text_column}'."}

        texts = session.dataframe[text_column].fillna("").tolist()

        # Convert ngram_range list to tuple for CountVectorizer
        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
            lowercase=lowercase
        )

        X = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        session.feature_matrix = X
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())

        n_rows, n_cols = X.shape
        non_zero = int(X.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "feature_names": feature_names
        }

        session.store_result("build_dtm_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """Aggregates and computes total term frequencies across all documents in the DTM."""
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found. Call build_dtm_tool first."}

        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        term_counts = {name: int(count) for name, count in zip(session.feature_names, counts)}

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "term_frequencies": term_counts
        }

        session.store_result("term_frequency_tool", {}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a heatmap of a slice of the raw Document-Term Matrix.

        Args:
            n_terms: Number of terms (columns) to include in the slice.
            n_documents: Number of documents (rows) to include in the slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found. Call build_dtm_tool first."}

        total_docs, total_terms = session.feature_matrix.shape
        doc_slice = min(n_documents, total_docs)
        term_slice = min(n_terms, total_terms)

        submatrix = session.feature_matrix[0:doc_slice, 0:term_slice].toarray()
        selected_terms = session.feature_names[0:term_slice]
        doc_labels = [f"Doc_{i}" for i in range(doc_slice)]

        fig, ax = plt.subplots(figsize=(max(6, term_slice * 0.8), max(4, doc_slice * 0.4)))
        sns.heatmap(
            submatrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=selected_terms,
            yticklabels=doc_labels,
            ax=ax
        )
        ax.set_title(f"DTM Heatmap (Top {doc_slice} Docs x Top {term_slice} Terms)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "sliced_shape": [doc_slice, term_slice],
            "terms": selected_terms,
            "documents": doc_labels,
            "matrix": submatrix.tolist()
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
