import pytest
import joblib
from testing import ARTIFACT_DIR


def test_data_transformation():
    pipeline = joblib.load(ARTIFACT_DIR / "model_pipeline.joblib")
    lob_categories = joblib.load(ARTIFACT_DIR / 'lob_categories.joblib')

    # Check if the pipeline and lob_categories are loaded correctly
    assert pipeline is not None, "Model pipeline should be loaded successfully."    
    assert lob_categories is not None, "LOB categories should be loaded successfully."
    
    