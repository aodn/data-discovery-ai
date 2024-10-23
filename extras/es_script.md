# ElasticSearch Script
## Find index which provided by IMOS
```
POST /es-indexer-edge/_search
{
  "size": 800,
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "providers.name": "IMOS"
          }
        }
      ]
    }
  }
}
```
## Find all records
```
POST /es-indexer-edge/_search
{
  "size": 11000,
  "query": {
    "match_all": {}
  }
}
```

References:
[1] [how do elastic search show all the hits for query](https://stackoverflow.com/questions/64871466/how-do-elastic-search-show-all-the-hits-for-query)