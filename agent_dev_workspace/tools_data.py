import os
import pathlib
from typing import Optional, List
import pandas as pd
import numpy as np
import nltk
from langchain_core.tools import tool
import matplotlib.pyplot as plt
import seaborn as sns


def make_tools(session):
    """
    Factory function returning the 7 data tools for data processing and analysis.
    """

    @tool
    def list_files_tool(subdirectory: str = ".") -> dict:
        """
        Lists directories and files relative to the repository root.

        Args:
            subdirectory: Subdirectory path relative to the repository root. Defaults to '.'.
        """
        if session.dataframe is None:
            pass  # list_files_tool does not depend on session.dataframe

        # Repo root is one directory up from the workspace root
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        target_dir = (repo_root / subdirectory).resolve()

        if not target_dir.exists() or not target_dir.is_dir():
            return {
                "error": f"Directory '{subdirectory}' does not exist relative to repository root."
            }

        directories = []
        files = []
        for entry in os.listdir(target_dir):
            if entry.startswith("."):
                continue
            full_path = target_dir / entry
            if full_path.is_dir():
                directories.append(entry)
            else:
                files.append(entry)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }
        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(
        start_index: int = 0,
        n_rows: int = 5,
        from_tail: bool = False,
        columns: Optional[List[str]] = None
    ) -> dict:
        """
        Inspects a flexible window of rows from session.dataframe (from top, middle, or tail).

        Args:
            start_index: Row index offset (supports positive or negative indexing). Defaults to 0.
            n_rows: Number of rows to inspect. Defaults to 5.
            from_tail: If True, inspects the last n_rows from the end of the DataFrame. Defaults to False.
            columns: Optional list of specific column names to view.
        """
        if session.dataframe is None:
            return {"error": "No DataFrame loaded in session. Please load a dataset first."}

        df = session.dataframe
        total_rows = len(df)

        if columns is not None:
            valid_cols = [c for c in columns if c in df.columns]
            if not valid_cols:
                return {"error": f"None of the requested columns {columns} exist in DataFrame."}
            slice_df = df[valid_cols]
        else:
            slice_df = df

        if from_tail:
            sliced = slice_df.tail(n_rows)
        else:
            if start_index < 0:
                start_idx = max(0, total_rows + start_index)
            else:
                start_idx = min(total_rows, start_index)
            
            end_idx = min(total_rows, start_idx + n_rows)
            sliced = slice_df.iloc[start_idx:end_idx]

        sample_rows = sliced.to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "total_cols": len(df.columns),
            "columns": list(slice_df.columns),
            "data_types": {c: str(df[c].dtype) for c in slice_df.columns},
            "sliced_indices": list(sliced.index),
            "n_rows_returned": len(sliced),
            "sample_rows": sample_rows,
        }
        session.store_result("inspect_data_tool", {
            "start_index": start_index,
            "n_rows": n_rows,
            "from_tail": from_tail
        }, summary)
        return summary

    @tool
    def check_missing_tool() -> dict:
        """
        Checks for missing, null, or empty/whitespace-only text values in session.dataframe.
        """
        if session.dataframe is None:
            return {"error": "No DataFrame loaded in session. Please load a dataset first."}

        df = session.dataframe
        total_rows = len(df)

        if "text" in df.columns:
            text_series = df["text"].astype(str)
            missing_text_mask = df["text"].isna() | (text_series.str.strip() == "") | (text_series == "None")
            missing_text_count = int(missing_text_mask.sum())
        else:
            missing_text_count = 0

        null_counts_per_column = df.isna().sum().to_dict()

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "missing_text_count": missing_text_count,
            "null_counts_per_column": {k: int(v) for k, v in null_counts_per_column.items()},
        }
        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(
        subset: Optional[str] = None,
        drop: bool = False,
        keep_first: bool = False
    ) -> dict:
        """
        Checks for duplicate entries in session.dataframe and optionally drops them.

        Args:
            subset: Column name to check for duplicates, or None to check original/data columns. Defaults to None.
            drop: Whether to drop duplicate rows in-place. Defaults to False.
            keep_first: If True, keeps the first occurrence when dropping (keep='first').
                       If False, drops all duplicate copies (keep=False). Defaults to False.
        """
        if session.dataframe is None:
            return {"error": "No DataFrame loaded in session. Please load a dataset first."}

        df = session.dataframe

        if subset is not None:
            if subset not in df.columns:
                return {"error": f"Column '{subset}' not found in session.dataframe."}
            subset_cols = [subset]
        else:
            # Exclude generated/pipeline columns when checking full-row duplicates
            generated_cols = {"unigrams", "text_length"}
            subset_cols = [col for col in df.columns if col not in generated_cols]

        initial_rows = len(df)

        # Full-row / subset duplicate detection
        dup_extra_mask = df.duplicated(subset=subset_cols, keep="first")
        n_duplicates = int(dup_extra_mask.sum())

        rows_dropped = 0
        if drop and n_duplicates > 0:
            keep_param = "first" if keep_first else False
            df.drop_duplicates(subset=subset_cols, keep=keep_param, inplace=True)
            rows_dropped = initial_rows - len(df)

        remaining_rows = len(session.dataframe)
        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "subset": subset,
            "n_duplicates": n_duplicates,
            "rows_dropped": rows_dropped,
            "remaining_rows": remaining_rows,
        }
        session.store_result("check_duplicates_tool", {"subset": subset, "drop": drop, "keep_first": keep_first}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 5, random_state: int = 42) -> dict:
        """
        Subsamples session.dataframe without discarding original row indices.

        Args:
            n: Number of rows to sample. Defaults to 5.
            random_state: Random seed for sampling. Defaults to 42.
        """
        if session.dataframe is None:
            return {"error": "No DataFrame loaded in session. Please load a dataset first."}

        df = session.dataframe
        sample_size = min(n, len(df))
        sampled_df = df.sample(n=sample_size, random_state=random_state)
        session.dataframe = sampled_df

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "sampled_rows": len(sampled_df),
            "sampled_indices": list(sampled_df.index),
            "random_state": random_state,
            "sample_records": sampled_df.to_dict(orient="records"),
        }
        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def tokenize_tool(text_column: str = "text") -> dict:
        """
        Tokenizes document text into lowercased unigrams using NLTK word_tokenize.

        Args:
            text_column: Column name containing text to tokenize. Defaults to 'text'.
        """
        if session.dataframe is None:
            return {"error": "No DataFrame loaded in session. Please load a dataset first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in session.dataframe."}

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        unigrams_list = []
        for val in df[text_column]:
            if pd.notna(val) and str(val).strip() != "":
                tokens = nltk.word_tokenize(str(val).lower())
            else:
                tokens = []
            unigrams_list.append(tokens)

        df["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        total_tokens = sum(len(toks) for toks in unigrams_list)
        avg_tokens = float(total_tokens / len(df)) if len(df) > 0 else 0.0

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "total_documents": len(df),
            "total_tokens": total_tokens,
            "average_tokens_per_doc": round(avg_tokens, 4),
            "sample_tokens": unigrams_list[:3],
        }
        session.store_result("tokenize_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def describe_data_tool() -> dict:
        """
        Calculates text length statistics and generates a boxplot by category_name.
        """
        if session.dataframe is None:
            return {"error": "No DataFrame loaded in session. Please load a dataset first."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Column 'text' not found in session.dataframe."}

        df["text_length"] = df["text"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        overall_stats = df["text_length"].describe().to_dict()
        overall_stats = {k: round(float(v), 4) for k, v in overall_stats.items()}

        per_category_stats = {}
        if "category_name" in df.columns:
            for cat, group in df.groupby("category_name"):
                cat_desc = group["text_length"].describe().to_dict()
                per_category_stats[str(cat)] = {k: round(float(v), 4) for k, v in cat_desc.items()}

        fig, ax = plt.subplots(figsize=(6, 4))
        if "category_name" in df.columns:
            sns.boxplot(x="category_name", y="text_length", data=df, ax=ax)
        else:
            sns.boxplot(y="text_length", data=df, ax=ax)

        ax.set_title("Document Text Length by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats,
        }
        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        tokenize_tool,
        describe_data_tool,
    ]
