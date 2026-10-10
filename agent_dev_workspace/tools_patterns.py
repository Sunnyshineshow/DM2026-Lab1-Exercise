import os
import tempfile
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# Import PAMI modules
from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: int = 1,
        k: int = 1
    ) -> Dict[str, Any]:
        """Mines frequent patterns per category using PAMI after vocabulary filtering.

        Args:
            category_name: The category label to filter documents by.
            filtering_method: 'variance', 'tfidf', or 'term_frequency'.
            algorithm: 'fpgrowth', 'topk', or 'maxfpgrowth'.
            min_sup: Minimum support count for fpgrowth/maxfpgrowth (must be >= 1).
            k: Top-k count for topk algorithm (must be >= 1).
        """
        # Validate inputs early
        valid_filtering = ["variance", "tfidf", "term_frequency"]
        if filtering_method not in valid_filtering:
            return {"error": f"Invalid filtering_method '{filtering_method}'. Must be one of {valid_filtering}."}

        valid_algorithms = ["fpgrowth", "topk", "maxfpgrowth"]
        if algorithm not in valid_algorithms:
            return {"error": f"Invalid algorithm '{algorithm}'. Must be one of {valid_algorithms}."}

        if algorithm in ["fpgrowth", "maxfpgrowth"] and min_sup < 1:
            return {"error": f"min_sup must be >= 1, got {min_sup}."}

        if algorithm == "topk" and k < 1:
            return {"error": f"k must be >= 1, got {k}."}

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        if session.dataframe is None:
            return {"error": "DataFrame is empty. Load a dataset first."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "DataFrame missing 'category_name' column."}

        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if len(cat_df) == 0:
            available = sorted(list(session.dataframe["category_name"].unique()))
            return {"error": f"Category '{category_name}' not found. Available categories: {available}."}

        texts = cat_df["text"].fillna("").tolist()

        # Safely fit CountVectorizer without raising ValueError on empty/stopword-only text
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X_sparse = vectorizer.fit_transform(texts)
            feature_names = list(vectorizer.get_feature_names_out())
        except ValueError:
            feature_names = []
            X_sparse = None

        if len(feature_names) == 0 or X_sparse is None:
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms in vocabulary"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        X = X_sparse.toarray()

        # Step 2: Apply chosen filtering_method
        if filtering_method == "variance":
            scores = np.var(X, axis=0, ddof=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            kept_mask = (scores > p5) & (scores < p95)
        elif filtering_method == "term_frequency":
            scores = X.sum(axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            kept_mask = (scores > p5) & (scores < p95)
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(X_sparse).toarray()
            scores = tfidf_matrix.mean(axis=0)
            p20 = np.percentile(scores, 20)
            kept_mask = scores >= p20

        kept_indices = np.where(kept_mask)[0]
        if len(kept_indices) == 0:
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        kept_terms = [feature_names[i] for i in kept_indices]
        X_filtered = X[:, kept_indices]

        # Step 3: Convert to binary transaction DataFrame
        binary_df = pd.DataFrame((X_filtered >= 1).astype(int), columns=kept_terms)
        binary_df = binary_df[binary_df.sum(axis=1) > 0]

        if len(binary_df) == 0:
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no non-empty transactions after filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 4: PAMI DF2DB conversion
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f_db:
            db_path = f_db.name

        try:
            obj = DF2DB.DF2DB(binary_df)
            obj.convert2TransactionalDatabase(
                db_path,
                condition=">=",
                thresholdValue=1
            )

            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(db_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = FAE.FAE(db_path, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(db_path, minSup=min_sup, sep="\t")

            if hasattr(miner, "mine"):
                miner.mine()
            elif hasattr(miner, "startMine"):
                miner.startMine()

            patterns_df = miner.getPatternsAsDataFrame()

            patterns = []
            if patterns_df is not None and len(patterns_df) > 0:
                for _, row in patterns_df.iterrows():
                    pat_str = str(row["Patterns"]).strip()
                    sup_val = int(row["Support"])
                    patterns.append({"pattern": pat_str, "support": sup_val})
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns),
            "patterns": patterns
        }

        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm
        }, summary)
        return summary

    return [mine_patterns_tool]
