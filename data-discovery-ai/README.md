# Data Discovery - AODN Portal AI Features
## Identified Tasks
### 1. Keywords Classification
#### Probelm Description

The new AODN Data Discovery portal is underpinned by a Geonetwork catalogue of metadata records that bring together well curated IMOS managed metadata records as well as records from external organisations and other contributors. 

IMOS records and any records that exist in the current AODN portal use keywords and vocabularies that are tightly controlled in order for the current portal facets to operate.  These are AODN vocabularies as defined in the ARDC repositories (https://vocabs.ardc.edu.au/).

Many other organisations use these AODN vocabularies and other well known vocabularies (e.g. GCMD) however there are many records in the catalogue that either use no keywords at all or keywords that are not from controlled vocabs.

The new AODN Data Discovery portal needs to filter metadata records based on a fixed set of “keywords” – using the AODN vocabularies.  The most important being 

- 'AODN Organisation Vocabulary',  

- 'AODN Instrument Vocabulary', 

- 'AODN Discovery Parameter Vocabulary'  

There are many metadata records that have no keywords or keywords that are not using a well known vocabularies. Given the mapping rules based on metadata records which have AODN vocabularies, we aim to develop a machine learning model to predict the AODN keywords for these uncategorised datasets, in order to provide suggestions that can be used by the Data Discovery portal or other applications.

Based on the query result from ElasticSearch, we currently identified **1075** over **9856** records which have no keywords (filed `_source.themes` is empty). These records are prepared as the target dataset that we are going to look after for this classification task.

Similarly, we also identified **1643** over **9856** records that have keywords using AODN vocabularies ('AODN Organisation Vocabulary', 'AODN Instrument Vocabulary', 'AODN Discovery Parameter Vocabulary'). This is the sample dataset we are going to work with to train the model. Two datasets are prepared for these records: one nested dataset which contains metadata title and description; and one flatterned dataset which contains keywords information.

A live example of an IMOS dataset [IMOS SOOP Underway Data from AIMS Vessel RV Solander Trip 6397 From 17 Mar 2016 To 23 Mar 2016](https://geonetwork-edge.edge.aodn.org.au/geonetwork/srv/eng/catalog.search#/metadata/52b58d9a-a0b4-4396-be8e-a9e5e2b493f0/formatters/xsl-view?root=div&view=advanced), which uses AODN vocabulary for keywords, is structured in our prepared dataset 'sample_AODN_vocabs_flattern' as follows:

**Preprocessed Dataset**

|id  | concept_id | concept_url | vocabolary|
----| ---- | ---- | ----|
52b58d9a-a0b4-4396-be8e-a9e5e2b493f0 | Practical salinity of the water body | http://vocab.nerc.ac.uk/collection/P01/current/PSLTZZ01 | AODN Discovery Parameter Vocabulary

We also prepared a dataset that contains title and description information, which is structured as follows:
| id | title | description | keywords|
----| ---- | ---- | ----|
0832b98c-602e-4902-8438-a80d402469ea | IMOS SOOP Underway Data from AIMS Vessel RV Cape Ferguson Trip 6321 From 30 Oct 2015 To 02 Nov 2015 | 'Ships of Opportunity' (SOOP) is a facility of the Australian 'Integrated Marine Observing System' (IMOS) project. This data set was collected by the SOOP sub-facility 'Sensors on Tropical Research Vessels' aboard the RV Cape Ferguson Trip 6321. | [{'concepts': [{'id': 'Practical salinity of the water body', 'url': 'http://vocab.nerc.ac.uk/collection/P01/current/PSLTZZ01'}, {'id': 'Temperature of the water body', 'url': 'http://vocab.nerc.ac.uk/collection/P01/current/TEMPPR01'}, {'id': 'Fluorescence of the water body', 'url': 'http://vocab.nerc.ac.uk/collection/P01/current/FLUOZZZZ'}, {'id': 'Turbidity of the water body', 'url': 'http://vocab.nerc.ac.uk/collection/P01/current/TURBXXXX'}], 'scheme': 'theme', 'description': '', 'title': 'AODN Discovery Parameter Vocabulary'}]

#### Acceptance Criteria
An Excel file which contains the predicted keywords suggestions for these non-categorised metadata records. 

### 2.Parameter Clustering

#### Problem Description
In metadata records (9856 items), there are only 3552 metadata records contains parameters (field `_source.summaries.parameter_vocabs` is not empty). We want to develop a machine learning model to predict the parameters for these uncategorised datasets, in order to provide suggestions that can be used by the Data Discovery portal or other applications.

#### Acceptance Criteria
An Excel file which contains the predicted parameter suggestions for these non-categorised metadata records.

A live example of parameters in IMOS datasets:
```
Parameters: {"['oxygen', 'chlorophyll', 'water pressure', 'turbidity', 'temperature', 'salinity']", "['air pressure', 'air-sea fluxes', 'wind', 'air temperature', 'humidity', 'precipitation and evaporation', 'water pressure', 'temperature']", "['wave']", "['air pressure', 'air-sea fluxes', 'wind', 'air temperature', 'humidity', 'precipitation and evaporation', 'water pressure', 'temperature', 'salinity']", "['salinity']", nan, "['ocean biota']", "['bathymetry', 'wave']", "['temperature', 'salinity']", "['water pressure', 'temperature', 'salinity']", "['ph (total scale) of the water body', 'oxygen', 'carbon', 'temperature', 'salinity']", "['air pressure', 'air-sea fluxes', 'wind', 'air temperature', 'humidity', 'water pressure', 'temperature']", "['current']", "['air pressure', 'air-sea fluxes', 'wind', 'air temperature', 'humidity', 'precipitation and evaporation', 'oxygen', 'chlorophyll', 'density', 'water pressure', 'turbidity', 'optical properties', 'temperature', 'salinity']", "['current', 'sea surface height']", "['ocean biota', 'water pressure', 'temperature']", "['temperature']", "['ocean biota', 'temperature', 'salinity']", "['water pressure', 'current', 'acoustics']", "['chlorophyll', 'turbidity', 'temperature', 'salinity']", "['air pressure', 'wind', 'carbon', 'water pressure', 'temperature', 'salinity']", "['air pressure', 'air-sea fluxes', 'wind', 'air temperature', 'humidity', 'precipitation and evaporation', 'water pressure', 'turbidity', 'temperature', 'salinity']", "['air pressure', 'wind', 'air temperature', 'precipitation and evaporation', 'chlorophyll', 'water pressure', 'turbidity', 'optical properties', 'temperature', 'salinity']", "['suspended particulate material', 'chlorophyll', 'other pigment', 'optical properties']", "['oxygen', 'chlorophyll', 'water pressure', 'turbidity', 'current', 'optical properties', 'temperature', 'salinity']", "['air-sea fluxes', 'optical properties']", "['ph (total scale) of the water body', 'oxygen', 'nutrient', 'chlorophyll', 'water pressure']", "['oxygen', 'nutrient', 'density', 'temperature', 'salinity']", "['chlorophyll', 'water pressure', 'temperature', 'salinity']", "['ph (total scale) of the water body', 'alkalinity', 'carbon', 'temperature', 'salinity']", "['water pressure', 'current', 'temperature', 'acoustics']", "['chlorophyll', 'optical properties', 'temperature', 'salinity']", "['acoustics']", "['wind', 'wave']", "['bathymetry']", "['water pressure', 'current', 'temperature', 'salinity']", "['water pressure', 'temperature']", "['chlorophyll']", "['air pressure', 'wind', 'air temperature', 'humidity', 'precipitation and evaporation', 'temperature']", "['air pressure', 'air-sea fluxes', 'uv radiation', 'wind', 'air temperature', 'humidity', 'precipitation and evaporation', 'water pressure', 'temperature']", "['wind']", "['wave', 'temperature']", "['air pressure', 'wind', 'air temperature', 'humidity', 'water pressure', 'temperature', 'salinity']", "['oxygen', 'chlorophyll', 'water pressure', 'current', 'optical properties', 'temperature', 'salinity']", "['wave', 'water pressure', 'temperature']", "['air pressure', 'air-sea fluxes', 'wind', 'air temperature', 'humidity', 'precipitation and evaporation', 'oxygen', 'nutrient', 'chlorophyll', 'wave', 'density', 'water pressure', 'turbidity', 'current', 'optical properties', 'temperature', 'salinity']", "['chlorophyll', 'other pigment', 'turbidity', 'optical properties', 'temperature', 'salinity']", "['current', 'temperature', 'salinity']"}
```

A prepared filtered dataset is stuctured as follows:
|id | title | description | parameter |
----| ---- | ---- | ---- |
52c92036-cea9-4b1a-b4f0-cc94b8b5df98 | IMOS - SRS - SST - L3C - NOAA 19 - 3 day - day time - Australia | This is a single sensor, multiple swath SSTskin product for three consecutive night-time periods, derived using observations from the AVHRR instrument on the NOAA-XX satellites. It is provided as a 0.02deg x 0.02deg cylindrical equidistant projected map over the region 70°E to 170°W, 20°N to 70°S.  Only day time passes are included in the product. Each grid cell contains the 3 day average of all the highest available quality SSTs that overlap with that cell, weighted by the area of overlap.  The diagram at  https://help.aodn.org.au/satellite-data/product-information/ indicates where this product fits within the GHRSST suite of NOAA/AVHRR products. Matchups with buoy SST observations (adjusted to skin depths) indicate typical 2014 biases of < 0.1 degC and standard deviations of 0.5 degC to 0.6 degC for NOAA-18 and NOAA-19.  Refer to the IMOS SST products web page at http://imos.org.au/sstproducts.html and Beggs et al. (2013) at http://imos.org.au/sstdata_references.html for further information. | ['temperature']

### 3. Searching Suggestions - Key Phrase Extraction
In the new AODN Data Discovery Portal, the searching suggestions are derived from an algorithm that is extracting the most common terms that appear in the title and abstract and storing this in ElasticSearch (field `_source.search_suggestions.abstract_phrases`). Some likely "non words" are stripped out, but there are still many unhelpful suggestions.

We aim to develop a machine learning model, to extract phrases that are more meaningful and targeted, so that can be used for searching suggestions. 

## Research Method
1. understand metadata
2. problem description
3. prepare datasets
4. research on methods
5. experiments
6. evaluation