import data_discovery_ai.utils.preprocessor as preprocessor
import data_discovery_ai.model.keywordModel as model
import data_discovery_ai.utils.es_connector as connector
import data_discovery_ai.service.keywordClassifier as keywordClassifier
import numpy as np
import json
import pandas as pd


def pipeline():
    # Step 0. load parameters
    # with open("data_discovery_ai/common/parameters.json", "r", encoding="utf-8") as f:
    #     params = json.load(f)
    # # # Step 1. connect elastic search
    # client = connector.connect_es(config_path="./esManager.config")

    # Step 2. conduct query
    # raw_data = connector.search_es(client)

    # Step 3. identify samples
    # vocabs = params["preprocessor"]["vocabs"]
    # labelledDS = preprocessor.identify_sample(raw_data, vocabs)
    # labelledDS = pd.read_csv("data_discovery_ai/input/keywords_sample.tsv", sep="\t")

    # # Step 4. preprocess sample data
    # preprocessed_samples = preprocessor.sample_preprocessor(labelledDS, vocabs)

    # # # Step 5. calculate embedding
    # sampleSet = preprocessor.calculate_embedding(preprocessed_samples)
    # # save as file for demo use
    # preprocessor.save_to_file(sampleSet, "data_discovery_ai/input/keyword_sample.pkl")

    # load sample set from file for demo use
    # sampleSet = preprocessor.load_from_file(
    #     "data_discovery_ai/input/keyword_sample.pkl"
    # )

    # # Step 6. prepare input X and output Y
    # X, Y, Y_df, labels = preprocessor.prepare_X_Y(sampleSet)
    # preprocessor.save_to_file(labels, "data_discovery_ai/input/labels.pkl")

    # # Step 7. identify rare labels
    # rare_label_threshold = params["preprocessor"]["rare_label_threshold"]
    # rare_label_index = preprocessor.identify_rare_labels(
    #     Y_df, rare_label_threshold, labels
    # )

    # # Step 8. custom resample X and Y only for records with rare labels
    # X_oversampled, Y_oversampled = preprocessor.resampling(
    #     X_train=X, Y_train=Y, strategy="custom", rare_keyword_index=rare_label_index
    # )

    # # # Step 9. Split X, Y into train and test sets
    # dimension, n_labels, X_train, Y_train, X_test, Y_test = (
    #     preprocessor.prepare_train_test(X_oversampled, Y_oversampled, params)
    # )

    # # # # Step 10. Resample train set with ROS
    # X_train_oversampled, Y_train_oversampled = preprocessor.resampling(
    #     X_train=X_train, Y_train=Y_train, strategy="ROS", rare_keyword_index=None
    # )

    # # Step 11. train model
    # # get class weight
    # label_weight_dict = model.get_class_weights(Y_train)
    # train keyword model
    # trained_model, history = model.keyword_model(
    #     X_train_oversampled,
    #     Y_train_oversampled,
    #     X_test,
    #     Y_test,
    #     label_weight_dict,
    #     dimension,
    #     n_labels,
    #     params,
    # )

    # trained_model = model.load_saved_model("pretrainedKeyword4demo")
    # # TODO: check the model is not None
    # # Step 12. Evaluate trained model
    # confidence = params["keywordModel"]["confidence"]
    # top_N = params["keywordModel"]["top_N"]
    # predicted_labels = model.prediction(X_test, trained_model, confidence, top_N)
    # eval = model.evaluation(Y_test=Y_test, predictions=predicted_labels)

    # Step 13. Prediction
    # example for use the trained model
    item_description = """
                        Ecological and taxonomic surveys of hermatypic scleractinian corals were carried out at approximately 100 sites around Lord Howe Island. Sixty-six of these sites were located on reefs in the lagoon, which extends for two-thirds of the length of the island on the western side. Each survey site consisted of a section of reef surface, which appeared to be topographically and faunistically homogeneous. The dimensions of the sites surveyed were generally of the order of 20m by 20m. Where possible, sites were arranged contiguously along a band up the reef slope and across the flat. The cover of each species was graded on a five-point scale of percentage relative cover. Other site attributes recorded were depth (minimum and maximum corrected to datum), slope (estimated), substrate type, total estimated cover of soft coral and algae (macroscopic and encrusting coralline). Coral data from the lagoon and its reef (66 sites) were used to define a small number of site groups which characterize most of this area.Throughout the survey, corals of taxonomic interest or difficulty were collected, and an extensive photographic record was made to augment survey data. A collection of the full range of form of all coral species was made during the survey and an identified reference series was deposited in the Australian Museum.In addition, less detailed descriptive data pertaining to coral communities and topography were recorded on 12 reconnaissance transects, the authors recording changes seen while being towed behind a boat.
                        The purpose of this study was to describe the corals of Lord Howe Island (the southernmost Indo-Pacific reef) at species and community level using methods that would allow differentiation of community types and allow comparisons with coral communities in other geographic locations.
                        """
    predicted_labels = keywordClassifier.keywordClassifier(
        trained_model="pretrainedKeyword4demo", description=item_description
    )

    print(predicted_labels)


if __name__ == "__main__":
    pipeline()
