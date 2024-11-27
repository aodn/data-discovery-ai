import pytest
from data_discovery_ai.pipeline import KeywordClassifierPipeline


def test_KeywordClassifierPipeline_init():
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


def test_make_prediction():
    pipeline = KeywordClassifierPipeline(
        isDataChanged=False, usePretrainedModel=True, model_name="development"
    )

    # set labels is None
    pipeline.set_labels(labels=None)
    # Test raise valueerror if with no labels
    with pytest.raises(ValueError, match=r"Prefined keywords should not be None"):
        pipeline.make_prediction(description="description")

