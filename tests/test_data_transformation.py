import pytest
import joblib
from pathlib import Path
from testing import ARTIFACT_DIR

@pytest.mark.skipif(
    not (ARTIFACT_DIR / "model_pipeline.joblib").exists(),
    reason="Artifact not available"
)
def test_data_transformation():
    pipeline = joblib.load(ARTIFACT_DIR / "model_pipeline.joblib")
    lob_categories = joblib.load(ARTIFACT_DIR / "lob_categories.joblib")

    assert pipeline is not None
    assert lob_categories is not None