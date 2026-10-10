import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools

def test_tools_patterns_all():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Known Answer #17: Variance filtering yields 0 terms for catA and catB
    var_catA_res = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "variance"})
    assert var_catA_res["n_patterns"] == 0
    assert var_catA_res["patterns"] == []
    assert var_catA_res["note"] == "no terms survived filtering"
    assert session.artifacts["patterns_catA_variance_fpgrowth"] == []

    var_catB_res = mine_patterns_tool.invoke({"category_name": "catB", "filtering_method": "variance"})
    assert var_catB_res["n_patterns"] == 0
    assert var_catB_res["patterns"] == []
    assert var_catB_res["note"] == "no terms survived filtering"
    assert session.artifacts["patterns_catB_variance_fpgrowth"] == []

    # Known Answer #17: Term-frequency filtering yields 0 terms for catA and catB
    tf_catA_res = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "term_frequency"})
    assert tf_catA_res["n_patterns"] == 0
    assert tf_catA_res["patterns"] == []
    assert tf_catA_res["note"] == "no terms survived filtering"

    tf_catB_res = mine_patterns_tool.invoke({"category_name": "catB", "filtering_method": "term_frequency"})
    assert tf_catB_res["n_patterns"] == 0
    assert tf_catB_res["patterns"] == []
    assert tf_catB_res["note"] == "no terms survived filtering"

    # Known Answer #17: TFIDF filtering with FPGrowth
    # catA -> alpha (support 3)
    catA_fpg_res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert catA_fpg_res["n_patterns"] == 1
    assert catA_fpg_res["patterns"][0]["pattern"] == "alpha"
    assert catA_fpg_res["patterns"][0]["support"] == 3
    assert session.artifacts["patterns_catA_tfidf_fpgrowth"] == catA_fpg_res["patterns"]

    # catB -> gamma (support 4)
    catB_fpg_res = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert catB_fpg_res["n_patterns"] == 1
    assert catB_fpg_res["patterns"][0]["pattern"] == "gamma"
    assert catB_fpg_res["patterns"][0]["support"] == 4

    # Known Answer #20: TopK algorithm (FAE) - catA and catB
    topk_catA_res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert topk_catA_res["n_patterns"] == 1
    assert topk_catA_res["patterns"][0]["pattern"] == "alpha"
    assert topk_catA_res["patterns"][0]["support"] == 3

    topk_catB_res = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert topk_catB_res["n_patterns"] == 1
    assert topk_catB_res["patterns"][0]["pattern"] == "gamma"
    assert topk_catB_res["patterns"][0]["support"] == 4

    # Known Answer #20: MaxFPGrowth returns 0 patterns for catA and catB
    max_catA_res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert max_catA_res["n_patterns"] == 0
    assert max_catA_res["patterns"] == []

    max_catB_res = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert max_catB_res["n_patterns"] == 0
    assert max_catB_res["patterns"] == []
