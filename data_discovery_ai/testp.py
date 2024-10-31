from data_discovery_ai.pipeline import KeywordClassifierPipeline


def pipeline(isDataChanged, usePretrainedModel, description, selected_model):
    keyword_classifier_pipeline = KeywordClassifierPipeline(
        isDataChanged=isDataChanged,
        usePretrainedModel=usePretrainedModel,
        model_name=selected_model,
    )
    if keyword_classifier_pipeline.usePretrainedModel:
        keyword_classifier_pipeline.make_prediction(description)
    else:
        if keyword_classifier_pipeline.isDataChanged:
            raw_data = keyword_classifier_pipeline.fetch_raw_data()
            sampleSet = keyword_classifier_pipeline.prepare_sampleSet(raw_data=raw_data)
        else:
            sampleSet = preprocessor.load_from_file("keyword_sample.pkl")
        train_test_data = keyword_classifier_pipeline.prepare_train_test_sets(sampleSet)
        keyword_classifier_pipeline.train_evaluate_model(train_test_data)

        keyword_classifier_pipeline.make_prediction(description)


def test():
    item_description = """
                        Ecological and taxonomic surveys of hermatypic scleractinian corals were carried out at approximately 100 sites around Lord Howe Island. Sixty-six of these sites were located on reefs in the lagoon, which extends for two-thirds of the length of the island on the western side. Each survey site consisted of a section of reef surface, which appeared to be topographically and faunistically homogeneous. The dimensions of the sites surveyed were generally of the order of 20m by 20m. Where possible, sites were arranged contiguously along a band up the reef slope and across the flat. The cover of each species was graded on a five-point scale of percentage relative cover. Other site attributes recorded were depth (minimum and maximum corrected to datum), slope (estimated), substrate type, total estimated cover of soft coral and algae (macroscopic and encrusting coralline). Coral data from the lagoon and its reef (66 sites) were used to define a small number of site groups which characterize most of this area.Throughout the survey, corals of taxonomic interest or difficulty were collected, and an extensive photographic record was made to augment survey data. A collection of the full range of form of all coral species was made during the survey and an identified reference series was deposited in the Australian Museum.In addition, less detailed descriptive data pertaining to coral communities and topography were recorded on 12 reconnaissance transects, the authors recording changes seen while being towed behind a boat.
                        The purpose of this study was to describe the corals of Lord Howe Island (the southernmost Indo-Pacific reef) at species and community level using methods that would allow differentiation of community types and allow comparisons with coral communities in other geographic locations.
                        """
    pipeline(
        isDataChanged=False,
        usePretrainedModel=False,
        description=item_description,
        selected_model="test_keyword_pipeline",
    )


if __name__ == "__main__":
    test()
