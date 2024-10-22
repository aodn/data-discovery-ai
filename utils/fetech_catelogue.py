from owslib.csw import CatalogueServiceWeb
import pandas as pd


# define item fields for use
record_format = {
    "identifier": "", # record.identifier, the id of a record
    "description": "", # record.identification.abstract, the description of an item
    "keywords": "" # the list of keywords in json format for each thesaurus {"title1":"title","keywords1":[keywords, keywords,...]}
}

# global varibales
current_position = 0
max_batch_size = 10


def record_process(record, record_format):
    rec = record_format
    rec['identifier'] = record.identifier
    try:
        item = record.identification[0]
        rec['description'] = item.abstract
        md_keywords = []
        if item.keywords:
            for mk in item.keywords:
                md_keywords.append({
                    'title': mk.thesaurus['title'],
                    'keywords': [md.name for md in mk.keywords]
                })
            rec['keywords'] = md_keywords
        else:
            rec['keywords'] = None
        return rec
    except Exception as e:
        pass


csw_url = "https://catalogue.aodn.org.au/geonetwork/srv/eng/csw?request=GetCapabilities&service=CSW&version=2.0.2"
csw = CatalogueServiceWeb(csw_url)
# print(csw.identification.type)
csw.getrecords2(
    outputschema="http://standards.iso.org/iso/19115/-3/mdb/2.0",
    esn="full",
    
    maxrecords=10
)

total_records = csw.results['matches']
print(total_records)

# for r in csw.records:
#     record = csw.records[r]
#     rec = record_process(record, record_format)
#     print(rec)

