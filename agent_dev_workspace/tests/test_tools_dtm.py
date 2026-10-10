import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_tools_dtm_all():
    # Single session setup reused across sequential pipeline operations
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load student-built tools directly
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Load test dataset
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    # Test build_dtm_tool (Known Answer #4)
    dtm_res = build_dtm_tool.invoke({"text_column": "text"})
    assert dtm_res["n_documents"] == 8
    assert dtm_res["n_terms"] == 5
    assert dtm_res["feature_names"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert dtm_res["non_zero"] == 21
    assert dtm_res["total_elements"] == 40
    assert dtm_res["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)
    assert "count_vectorizer" in session.artifacts

    # Test term_frequency_tool (Known Answer #14)
    tf_res = term_frequency_tool.invoke({})
    freqs = tf_res["term_frequencies"]
    assert freqs["alpha"] == 6
    assert freqs["always"] == 7
    assert freqs["beta"] == 3
    assert freqs["delta"] == 4
    assert freqs["gamma"] == 7

    # Test dtm_heatmap_tool (Known Answer #21)
    heatmap_res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert heatmap_res["sliced_shape"] == [3, 2]
    assert heatmap_res["terms"] == ["alpha", "always"]
    assert heatmap_res["matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert session.pending_figure is not None
