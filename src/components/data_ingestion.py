import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path  
from src.constant import *
from src.exception import Custom_exception
from src.logger import logging
from src.utils import main_utils

from dataclasses import dataclass 

@dataclass
class DataIngestionConfig:
    artifact_folder=os.path.join(artifact_folder)
    

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.utils=main_utils()
        
    def export_collection_as_dataframe(self,collection_name,db_name):
        
        try:
            mongo_client=MongoClient(MONGO_DB_URL)
            collection=mongo_client[db_name][collection_name]
            
            df=pd.DataFrame(list(collection.find()))
            
            if '_id' in df.columns.to_list():
                df=df.drop('_id',axis=1)
                
            df.replace({'na':np.nan},inplace=True)
            
            return df
        except Exception as e:
            raise Custom_exception(e,sys)
        
        
    def export_data_into_features_store_file_path(self)-> pd.DataFrame:
        try:
            logging.info('exporting data from MongoDb')
            raw_file_path=self.data_ingestion_config.artifact_folder
            
            os.makedirs(raw_file_path,exist_ok=True)
            
            sensored_data=self.export_collection_as_dataframe(
                MONGO_COLLECTION_NAME,MONGO_DATABASE_NAME
                )
            
            logging.info(f'saving exported data into featre store file path :{raw_file_path}')
            feature_store_file_path=os.path.join(raw_file_path,'wafers_fault.csv')
            
            sensored_data.to_csv(feature_store_file_path,index=False)
            
            return feature_store_file_path
        
        except Exception as e:
            raise Custom_exception(e,sys)
        
        
    def initiate_data_ingestion(self)-> Path:
        logging.info('Entered initiated_data_ingestrion method of data_integration class')
        
        try:
            feature_store_file_path=self.export_data_into_features_store_file_path()
            
            logging.info('GOT THE DATA from mongodb')
            logging.info('exited initiatr_data_ingestion method of data ingestion class')
            
            return feature_store_file_path
        except Exception as e:
            raise Custom_exception(e,sys)
            
            
            
            
            
             
         
            
            
        
    
