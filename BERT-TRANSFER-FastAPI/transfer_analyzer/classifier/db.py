import pymongo
CONNECTION_STRING = ""
client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('transfer')



    
