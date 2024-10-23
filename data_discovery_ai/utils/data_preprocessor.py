import json
import pandas as pd
import csv
import re
import owslib


# Read search result from ES, which is copied from the console of ES and saved in a json file. Convert it to tsv
def json2tsv(input, output):
    with open(f"./input/{input}", "r", encoding="utf-8") as f:
        json_string = f.read()
        data = json.loads(json_string)
        indexes = data["hits"]["hits"]
    df = pd.json_normalize(indexes)

    # df.columns = [c.split("_")[1] for c in list(df.columns)]
    print(df.columns)
    # df = df.map(lambda x: x.replace('\n', ' ').replace('\\n', ' ').replace(',', ' ') if isinstance(x, str) else x)
    df.to_csv(f"./output/{output}", index=False, sep="\t")


# Explore dataset
def explore_dataset(ds):
    print(f"Columns: {ds.columns} \n")
    print(f"Shape: {ds.shape} \n")
    print(ds.head)


def keywords_df(ds):
    keywords = ds[["_id", "_source.title", "_source.description", "_source.themes"]]
    keywords.columns = ["id", "title", "description", "keywords"]
    # print(keywords.head)

    keywords.loc[:, "keywords"] = keywords["keywords"].apply(lambda k: eval(k))

    # Step 1: identify target dataset: metadata has no keywords record
    # filtered = keywords[keywords['keywords'].apply(lambda x: len(x) == 0)]
    # filtered.loc[filtered['keywords'].apply(lambda x: len(x) == 0), 'keywords'] = None
    # ds_to_tsv(filtered, "no_keywords.tsv")

    # Step 2: identify sample dataset: metadata keywords record uses AODN vocabulary
    vocabs = [
        "AODN Organisation Vocabulary",
        "AODN Instrument Vocabulary",
        "AODN Discovery Parameter Vocabulary",
        "AODN Platform Vocabulary",
        "AODN Parameter Category Vocabulary",
    ]
    # vocabs = ['AODN Discovery Parameter Vocabulary']
    sample = keywords[
        keywords["keywords"].apply(
            lambda terms: any(
                any(vocab in k["title"] for vocab in vocabs) for k in terms
            )
        )
    ]

    # Step 3: flattern sample table to get keywords table
    result = pd.concat(
        sample.apply(lambda row: flattern_keywords(row), axis=1).tolist(),
        ignore_index=True,
    )
    filter_result = result[result["vocabulary"].isin(vocabs)]

    # row = sample[sample['id'] == "52c92036-cea9-4b1a-b4f0-cc94b8b5df98"]
    # rowdf = flattern_keywords(row)
    ds_to_tsv(sample, "AODN_parameter_vocabs.tsv")
    ds_to_tsv(filter_result, "AODN_parameter_vocabs_flattern.tsv")


def flattern_keywords(row):
    id = []
    concept_id = []
    concept_url = []
    vocabolary = []

    keywords = row["keywords"]
    for k in keywords:
        concept = k.get("concepts")
        # print(k)
        for c in concept:
            id.append(row["id"])
            vocabolary.append(k.get("title"))
            if c["id"] is not None:
                concept_id.append(c["id"])
            try:
                concept_url.append(c["url"])
            except KeyError as e:
                concept_url.append(None)

    return pd.DataFrame(
        {
            "id": id,
            "concept_id": concept_id,
            "concept_url": concept_url,
            "vocabulary": vocabolary,
        }
    )


def parameter_df(ds):
    ds = ds[
        [
            "_id",
            "_source.title",
            "_source.description",
            "_source.summaries.parameter_vocabs",
        ]
    ]
    ds.columns = ["id", "title", "description", "parameter"]
    ds = ds.dropna(subset=["parameter"])
    print(ds.shape)
    return ds


def ds_to_tsv(df, file_name):
    df.to_csv(f"./output/{file_name}", index=False, sep="\t")


"""
    Explore json fields
"""


def explore_jsonDS(jsonFile):
    with open(f"./input/{jsonFile}", "r", encoding="utf-8") as finput:
        data = json.load(finput)
    hits = data["hits"]["hits"][0]
    print(hits)
    with open("./output/sample_test.json", "w") as foutput:
        json.dump(hits, foutput, indent=4)


"""
Parse description phrases
"""


def description_phrases():
    ds = pd.read_csv("./output/AODN.tsv", sep="\t")
    ds = ds[["_id", "_source.title", "_source.description"]]
    ds_to_tsv(ds, file_name="AODN_description.tsv")


if __name__ == "__main__":
    json2tsv("es_searchAll_result.json", "AODN.tsv")

    # ds = pd.read_csv("./output/AODN.tsv", sep='\t')
    # keywords_df(ds)
    # ds = pd.read_csv("./output/AODN.tsv", sep="\t")
    # explore_dataset(ds)

    # description_phrases()
