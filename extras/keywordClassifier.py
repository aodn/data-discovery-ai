import utils.preprocessor as preprocessor
import model.keywordModel as model
import numpy as np

VOCABS = ['AODN Instrument Vocabulary', 'AODN Discovery Parameter Vocabulary', 'AODN Platform Vocabulary']


def keywordClassifier():
    # load sample set
    sampleDS = preprocessor.load_sample()
    # load target set
    targetDS = preprocessor.load_target()

    # preprocess sample set
    # reformat labels
    sampleDS = preprocessor.extract_labels(sampleDS, VOCABS)
    # remove empty value records
    list_lengths = sampleDS['keywords'].apply(len)
    empty_keywords_records_index= list_lengths[list_lengths == 0].index.tolist()
    empty_keywords_records = []
    for index in empty_keywords_records_index:
        empty_keywords_records.append(sampleDS.iloc[index]['id'])
    empty_keywords_records
    sampleDS_cleaned = sampleDS[~sampleDS['id'].isin(empty_keywords_records)]

    # sampleDS_cleaned = sampleDS

    # prepare input X matrix
    X = np.array(sampleDS_cleaned['embedding'].tolist())

    # prepare output Y matrix
    Y = preprocessor.prepare_Y_matrix(sampleDS_cleaned)
    
    # save label maps
    keywords_set = Y.columns.to_list()

    # identify rare labels
    keyword_distribution = Y.copy()
    keyword_distribution = keyword_distribution.sum()
    keyword_distribution.sort_values()
    keyword_distribution_df = keyword_distribution.to_frame(name='count')
    threshold = 10
    rare_keyword = keyword_distribution_df[keyword_distribution_df['count'] <= threshold]
    rare_keyword = rare_keyword.index.to_list()
    rare_keyword_index = []
    for item in rare_keyword:
        if item in keywords_set:
            index_in_keywords_set = keywords_set.index(item)
            rare_keyword_index.append(index_in_keywords_set)
    # argument recrods with rare labels
    Y = Y.to_numpy()
    X_oversampled, Y_oversampled = preprocessor.resampling(X_train=X, Y_train=Y, strategy='custom', rare_keyword_index=rare_keyword_index)

    # # split train and test set
    # X_oversampled, Y_oversampled = X, Y
    dimension, n_labels, X_train, Y_train, X_test, Y_test = preprocessor.prepare_train_validation_test(X_oversampled, Y_oversampled)

    # # oversample train set
    X_train_oversampled, Y_train_oversampled = X_train, Y_train
    X_train_oversampled, Y_train_oversampled = preprocessor.resampling(X_train=X_train, Y_train=Y_train, strategy='ROS', rare_keyword_index=None)
    # X_train_oversampled, Y_train_oversampled = preprocessor.resampling(X_train=X_train, Y_train=Y_train, strategy='SMOTE', rare_keyword_index=None)

    # # get label weight label:{0:weight for negative label, 1:weight for positive label}
    label_weight_dict ={}
    label_weight_dict = model.get_class_weights(Y_train)
    trained_model, history = model.keyword_model(X_train_oversampled, Y_train_oversampled, X_test, Y_test, label_weight_dict, dimension, n_labels)
    # trained_model = load_model("output/saved/best-trained-keyword.keras", compile=False)


    # evaluate model
    confidence = 0.4
    top_N = 2
    predicted_labels = model.prediction(X_test, trained_model, confidence, top_N)
    eval = model.evaluation(Y_test=Y_test, predictions=predicted_labels)

    # prediction
    target_X = np.array(targetDS['embedding'].tolist())
    target_predicted_labels = model.prediction(target_X, trained_model, confidence, top_N)
    model.get_predicted_keywords(target_predicted_labels, keywords_set, targetDS)


if __name__ == "__main__":
    # preprocessor.load_datasets_from_source(VOCABS)
    keywordClassifier()