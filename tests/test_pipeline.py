import pytest
from data_discovery_ai.pipeline import KeywordClassifierPipeline


def test_keyword_classifier_pipeline_init():
    """Test the init function of KeywordClassifierPipeline."""
    # Test init function with valid model name
    pipeline = KeywordClassifierPipeline(
        isDataChanged=False, usePretrainedModel=True, model_name="development"
    )

    # Assertions
    assert pipeline.isDataChanged is False
    assert pipeline.usePretrainedModel is True
    assert pipeline.model_name == "development"
    assert pipeline.labels is None

    # Test invalid model name
    with pytest.raises(ValueError, match=r"Available model name: .*"):
        KeywordClassifierPipeline(
            isDataChanged=False, usePretrainedModel=True, model_name="invalid_model"
        )
