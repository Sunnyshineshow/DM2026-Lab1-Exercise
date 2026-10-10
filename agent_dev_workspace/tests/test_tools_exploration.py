import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def test_tools_exploration_all():
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

    # Load exploration tools
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    feature_correlation_matrix_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # 1. Test cosine_similarity_tool (Known Answer #16)
    # Doc 0 vs 1: 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)
    assert len(res_0_1["term_breakdown"]) > 0

    # Doc 0 vs 3: 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)

    # Doc 3 vs 4: 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Doc 0 vs 7 (empty text): 0.0
    res_0_7 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["similarity"] == 0.0

    # Test out of bounds index check
    oob_res = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 99})
    assert "error" in oob_res
    assert "out of bounds" in oob_res["error"]

    # 2. Test feature_correlation_matrix_tool (Known Answer #19)
    corr_res = feature_correlation_matrix_tool.invoke({})
    assert corr_res["n_terms"] == 5
    assert corr_res["feature_names"] == ["alpha", "gamma", "delta", "beta", "always"]
    assert session.pending_figure is not None

    matrix = np.array(corr_res["correlation_matrix"])
    
    # Assert diagonals are 1.0 (since no terms in fixture have zero variance)
    assert np.allclose(np.diag(matrix), 1.0)

    # Assert key feature correlations from Known Answer #19
    # alpha (idx 0) vs beta (idx 3) = 0.8885
    assert matrix[0, 3] == pytest.approx(0.8885, abs=1e-3)
    assert matrix[3, 0] == pytest.approx(0.8885, abs=1e-3)

    # gamma (idx 1) vs delta (idx 2) = 0.8307
    assert matrix[1, 2] == pytest.approx(0.8307, abs=1e-3)
    assert matrix[2, 1] == pytest.approx(0.8307, abs=1e-3)

    # alpha (idx 0) vs delta (idx 2) = -0.6882
    assert matrix[0, 2] == pytest.approx(-0.6882, abs=1e-3)

    # alpha (idx 0) vs always (idx 4) = 0.2601
    assert matrix[0, 4] == pytest.approx(0.2601, abs=1e-3)
