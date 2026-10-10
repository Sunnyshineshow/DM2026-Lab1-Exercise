import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_tools_filtering_all():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load DTM tools and build matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({"text_column": "text"})

    # Load filtering tools
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Test variance_filter_tool (Known Answer #5)
    var_res = variance_filter_tool.invoke({"threshold": 0.15})
    assert var_res["n_kept"] == 4
    assert set(var_res["kept_features"]) == {"alpha", "beta", "delta", "gamma"}
    assert var_res["removed_features"] == ["always"]
    assert var_res["variances"]["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert var_res["variances"]["always"] == pytest.approx(0.109375, abs=1e-3)
    assert var_res["variances"]["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert var_res["variances"]["delta"] == pytest.approx(0.25, abs=1e-3)
    assert var_res["variances"]["gamma"] == pytest.approx(1.109375, abs=1e-3)

    # Test pearson_filter_tool (Known Answer #6)
    pearson_res = pearson_filter_tool.invoke({"target_class": "catB"})
    p_corrs = pearson_res["correlations"]
    assert p_corrs["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert p_corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert p_corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert p_corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert p_corrs["gamma"] == pytest.approx(0.8307, abs=1e-3)

    # Test pearson_filter_tool invalid class check
    p_invalid_res = pearson_filter_tool.invoke({"target_class": "nonexistent_class"})
    assert "error" in p_invalid_res
    assert "not found in labels" in p_invalid_res["error"]

    # Test spearman_filter_tool (Known Answer #6)
    spearman_res = spearman_filter_tool.invoke({"target_class": "catB"})
    s_corrs = spearman_res["correlations"]
    assert s_corrs["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert s_corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert s_corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert s_corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert s_corrs["gamma"] == pytest.approx(0.9363, abs=1e-3)

    # Test spearman_filter_tool invalid class check
    s_invalid_res = spearman_filter_tool.invoke({"target_class": "nonexistent_class"})
    assert "error" in s_invalid_res
    assert "not found in labels" in s_invalid_res["error"]

    # Test mismatched label count check for both Pearson and Spearman
    original_labels = session.labels
    session.labels = original_labels[:-1]  # Drop one label to cause mismatch

    p_mismatch_res = pearson_filter_tool.invoke({"target_class": "catB"})
    assert "error" in p_mismatch_res
    assert "does not match DTM row count" in p_mismatch_res["error"]

    s_mismatch_res = spearman_filter_tool.invoke({"target_class": "catB"})
    assert "error" in s_mismatch_res
    assert "does not match DTM row count" in s_mismatch_res["error"]

    session.labels = original_labels  # Restore
