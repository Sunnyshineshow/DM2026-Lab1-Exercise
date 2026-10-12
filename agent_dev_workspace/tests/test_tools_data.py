import os
import pathlib
import pytest
import pandas as pd
import numpy as np

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def create_test_session_and_tools():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    tools = make_tools(session)
    tools_by_name = {t.name: t for t in tools}
    
    # repo_root is 3 levels up from agent_dev_workspace/tests/test_tools_data.py
    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    fixture_path = str(repo_root / "agent_dev" / "sample_fixture.csv")
    
    load_res = load_dataset_tool.invoke({
        "file_path": fixture_path,
        "text_column": "text",
        "label_column": "label"
    })
    return session, tools_by_name


def test_check_missing_tool():
    session, tools = create_test_session_and_tools()
    check_missing_tool = tools["check_missing_tool"]
    
    res = check_missing_tool.invoke({})
    # Known answer #2: exactly 1 row has missing/empty text (the last row)
    assert res["missing_text_count"] == 1


def test_check_duplicates_tool():
    session, tools = create_test_session_and_tools()
    check_duplicates_tool = tools["check_duplicates_tool"]
    tokenize_tool = tools["tokenize_tool"]
    
    # Run tokenization first to add 'unigrams' column (pipeline-generated column)
    tokenize_tool.invoke({"text_column": "text"})
    
    # Full-row check (subset=None) should exclude generated columns like 'unigrams'
    # Known answer #3: row 6 is exact full-row duplicate of row 5 ("always gamma delta", catB) -> n_duplicates=1
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check["n_duplicates"] == 1
    
    # Test that identical text with different labels isn't treated as a full-row duplicate.
    # Keep all category-related fields consistent ('label', 'category_name', 'category'):
    test_df = session.dataframe.copy()
    extra_row = test_df.iloc[2].copy()  # Original row 2: "always alpha beta", catA
    extra_row["label"] = "catB"
    extra_row["category_name"] = "catB"
    if "category" in test_df.columns:
        extra_row["category"] = 1
        
    test_df = pd.concat([test_df, pd.DataFrame([extra_row])], ignore_index=True)
    session.dataframe = test_df
    
    # Checking full-row duplicates on test_df:
    # Row 2 ("always alpha beta", catA) vs extra_row ("always alpha beta", catB) have DIFFERENT labels,
    # so full-row check (subset=None) should NOT count them as duplicates.
    res_diff_label = check_duplicates_tool.invoke({"subset": None, "drop": False})
    assert res_diff_label["n_duplicates"] == 1
    
    # Explicitly checking subset="text" WOULD find the duplicate text (rows 2 & extra_row, plus 5 & 6) -> 2 duplicate pairs
    res_text_subset = check_duplicates_tool.invoke({"subset": "text", "drop": False})
    assert res_text_subset["n_duplicates"] == 2

    # Reset session and verify dropping with keep=False drops rows 5 and 6, leaving 6 rows
    session, tools = create_test_session_and_tools()
    res_drop = tools["check_duplicates_tool"].invoke({"drop": True})
    assert res_drop["rows_dropped"] == 2
    assert res_drop["remaining_rows"] == 6


def test_list_files_tool():
    session, tools = create_test_session_and_tools()
    list_files_tool = tools["list_files_tool"]
    
    # Known answer #11: subdirectory="newdataset" returns files: ["Reddit-stock-sentiment.csv"]
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert "Reddit-stock-sentiment.csv" in res["files"]
    
    # Test path outside repository: test against an existing directory outside repo
    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    assert repo_root.exists()
    
    res_outside = list_files_tool.invoke({"subdirectory": ".."})
    assert "error" in res_outside or res_outside.get("subdirectory") == ".."


def test_sample_data_tool():
    session, tools = create_test_session_and_tools()
    sample_data_tool = tools["sample_data_tool"]
    
    # Known answer #13: n=4, random_state=42 returns rows [1, 5, 0, 7]
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res["sampled_rows"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]


def test_tokenize_tool():
    session, tools = create_test_session_and_tools()
    tokenize_tool = tools["tokenize_tool"]
    
    # Known answer #10: tokenized output for all 8 rows
    res = tokenize_tool.invoke({"text_column": "text"})
    assert res["total_documents"] == 8
    
    unigrams = session.artifacts["unigrams"]
    assert len(unigrams) == 8
    
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]
    
    for i, expected in enumerate(expected_tokens):
        assert unigrams[i] == expected, f"Row {i} tokens mismatch: got {unigrams[i]}, expected {expected}"


def test_describe_data_tool():
    session, tools = create_test_session_and_tools()
    describe_data_tool = tools["describe_data_tool"]
    
    # Known answer #18: text length stats over sample_fixture.csv
    res = describe_data_tool.invoke({})
    stats = res["overall_stats"]
    
    assert stats["count"] == 8
    assert np.isclose(stats["mean"], 19.875, atol=1e-3)
    assert np.isclose(stats["min"], 0.0, atol=1e-3)
    assert np.isclose(stats["max"], 30.0, atol=1e-3)
    
    # Check pending figure is set
    assert session.pending_figure is not None


def test_inspect_data_tool():
    session, tools = create_test_session_and_tools()
    inspect_data_tool = tools["inspect_data_tool"]
    
    # 1. Basic inspection
    res = inspect_data_tool.invoke({"n_rows": 3})
    assert res["total_rows"] == 8
    assert res["total_cols"] >= 2
    assert "text" in res["columns"]
    assert "category_name" in res["columns"]
    
    sample_rows = res["sample_rows"]
    assert len(sample_rows) == 3
    assert sample_rows[0]["text"] == "always alpha alpha alpha beta"
    assert sample_rows[0]["category_name"] == "catA"

    # 2. Large row bounds test
    res_large = inspect_data_tool.invoke({"n_rows": 100})
    assert len(res_large["sample_rows"]) == 8

    # 3. Middle window inspection (start_index = 3, n_rows = 2)
    res_mid = inspect_data_tool.invoke({"start_index": 3, "n_rows": 2})
    assert res_mid["sliced_indices"] == [3, 4]
    assert len(res_mid["sample_rows"]) == 2

    # 4. Tail inspection (from_tail = True)
    res_tail = inspect_data_tool.invoke({"n_rows": 3, "from_tail": True})
    assert len(res_tail["sample_rows"]) == 3
    assert res_tail["sliced_indices"] == [5, 6, 7]

    # 5. Column filtering
    res_cols = inspect_data_tool.invoke({"n_rows": 2, "columns": ["text"]})
    assert res_cols["columns"] == ["text"]
    assert "category_name" not in res_cols["sample_rows"][0]
