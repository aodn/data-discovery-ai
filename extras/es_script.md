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

## Find all records with paginate search
1. the initial search by sort
```
POST /es-indexer-staging/_search
{
  "query": {
    "match_all": {}
  },
  "sort": [
        {
            "summaries.creation": "asc"
        }
    ]
}
```
2. set pit
```
POST /es-indexer-staging/_pit?keep_alive=1m
```

3. search with `search_after`

References:
[1] [how do elastic search show all the hits for query](https://stackoverflow.com/questions/64871466/how-do-elastic-search-show-all-the-hits-for-query)
