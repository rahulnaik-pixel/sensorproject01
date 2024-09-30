from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri='mongodb+srv://rahulmoto520:zAtWQNj67hEjihdp@cluster0.thvfu.mongodb.net/?retryWrites=true&w=majority'

#create a new client and connect a server
client=MongoClient(uri)

#create a database and collection name
DATABASE_NAME="pwskills"
COLLECTION_NAME="waferfault"

df=pd.read_csv('C:\Users\rahul\Downloads\Telegram Desktop\newmlprojec\notebooks\wafer_23012020_041211.csv')

df=df.drop('Unnamed: 0',axis=1)

json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

