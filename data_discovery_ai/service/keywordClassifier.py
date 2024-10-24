import data_discovery_ai.utils.preprocessor as preprocessor
import data_discovery_ai.model.keywordModel as model


# TODO: method's param needs type hinting. e.g name: str; Optional[...], Union etc.
def keywordClassifier(trained_model, description, params, labels, dim):
    target_X = preprocessor.get_description_embedding(description).reshape(1, dim)
    target_predicted_labels = model.prediction(
        target_X,
        trained_model,
        params["keywordModel"]["confidence"],
        params["keywordModel"]["top_N"],
    )
    return model.get_predicted_keywords(target_predicted_labels, labels)
