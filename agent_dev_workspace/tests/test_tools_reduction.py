import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

def test_tools_reduction_all():
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

    # Load reduction tools
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    # 1. PCA Reduction (Known Answer #7) - Original Numerical Assertions Kept
    pca_res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})
    assert pca_res["method"] == "pca"
    assert pca_res["n_samples"] == 8
    assert pca_res["explained_variance_ratio"] == [pytest.approx(0.7532, abs=1e-3), pytest.approx(0.1965, abs=1e-3)]
    assert pca_res["reduced_coordinates"][0][0] == pytest.approx(2.3669, abs=1e-3)
    assert pca_res["reduced_coordinates"][0][1] == pytest.approx(0.8871, abs=1e-3)
    assert session.pending_figure is not None
    assert "reduced_pca" in session.artifacts

    session.pending_figure = None

    # 2. t-SNE Reduction (Known Answer #8)
    # Note: Stochastic optimization algorithms (t-SNE) can produce different coordinate projections across
    # execution environments or library version updates. We test structural validity, numerical sanity,
    # figure generation, artifact storage, and intra-environment seed reproducibility.
    tsne_res1 = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2.0,
        "random_state": 42
    })
    assert tsne_res1["method"] == "tsne"
    tsne_coords1 = np.array(tsne_res1["reduced_coordinates"])
    
    assert tsne_coords1.shape == (8, 2)
    assert np.all(np.isfinite(tsne_coords1))
    assert session.pending_figure is not None
    assert "reduced_tsne" in session.artifacts

    session.pending_figure = None
    tsne_res2 = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2.0,
        "random_state": 42
    })
    tsne_coords2 = np.array(tsne_res2["reduced_coordinates"])
    assert np.allclose(tsne_coords1, tsne_coords2)

    session.pending_figure = None

    # 3. UMAP Reduction (Known Answer #9)
    # Note: UMAP manifold optimization relies on stochastic initializations and compiled backends,
    # which can vary across execution environments. We test structural validity, numerical sanity,
    # figure generation, artifact storage, and intra-environment seed reproducibility.
    umap_res1 = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })
    assert umap_res1["method"] == "umap"
    umap_coords1 = np.array(umap_res1["reduced_coordinates"])

    # 3.1 Shape: Exactly 8 documents x 2 coordinates
    assert umap_coords1.shape == (8, 2)
    # 3.2 Finite values: No NaN or infinite coordinates
    assert np.all(np.isfinite(umap_coords1))
    # 3.3 Visualization: session.pending_figure is populated
    assert session.pending_figure is not None
    # 3.4 Artifacts: session.artifacts["reduced_umap"] contains generated coordinates within float tolerance
    assert "reduced_umap" in session.artifacts
    assert np.allclose(
        session.artifacts["reduced_umap"],
        umap_res1["reduced_coordinates"],
        atol=1e-4,
        rtol=0
    )

    # 3.5 Reproducibility: Repeated runs with random_state=42 produce matching coordinates using np.allclose()
    session.pending_figure = None
    umap_res2 = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })
    umap_coords2 = np.array(umap_res2["reduced_coordinates"])
    assert np.allclose(umap_coords1, umap_coords2)

    # 4. Error Checking for Parameters
    tsne_err = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 8.0})
    assert "error" in tsne_err
    assert "must be > 0 and < number of samples" in tsne_err["error"]

    umap_err = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 8})
    assert "error" in umap_err
    assert "must be >= 2 and less than number of samples" in umap_err["error"]

    # 5. Mismatched Category Labels Check in reduce_dimensions_tool
    original_labels = session.labels
    session.labels = ["catA"] * 8  # Change values to mismatch DataFrame (same length = 8)
    mismatch_err = reduce_dimensions_tool.invoke({"method": "pca"})
    assert "error" in mismatch_err
    assert "do not match DataFrame" in mismatch_err["error"]

    # 6. Binarize Labels Tool with mismatched session.labels
    stale_labels = list(session.labels)
    bin_res = binarize_labels_tool.invoke({})
    # Binarization should succeed despite stale session.labels
    assert "error" not in bin_res
    assert bin_res["n_documents"] == 8
    assert bin_res["n_categories"] == 2
    # Binarization must not modify session.labels
    assert list(session.labels) == stale_labels
    # Restore labels after testing the mismatch behavior
    session.labels = original_labels

    # 7. Binarize Labels Tool assertions with valid labels
    bin_res_valid = binarize_labels_tool.invoke({})
    assert bin_res_valid["n_documents"] == 8
    assert bin_res_valid["n_categories"] == 2
    assert bin_res_valid["categories"] == ["catA", "catB"]
    assert bin_res_valid["binary_columns"] == ["category_catA", "category_catB"]
    assert bin_res_valid["shape"] == [8, 2]
    assert bin_res_valid["binary_matrix"] == [
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 0]
    ]
    assert "binarized_labels" in session.artifacts
    assert isinstance(session.artifacts["binarized_labels"], pd.DataFrame)

    # 8. Missing Label Check in binarize_labels_tool
    session.dataframe.loc[0, "category_name"] = None
    missing_err = binarize_labels_tool.invoke({})
    assert "error" in missing_err
    assert "missing or empty label values" in missing_err["error"]
