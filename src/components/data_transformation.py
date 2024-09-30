import sys
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.constant import *
from src.logger import logging
from src.utils.main_utils import MainUtils
from src.exception import Custom_exception
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifact_dir=os.path.join(artifact_folder)
    transformed_train_file_path=os.path.join(artifact_dir,'train.npy')
    transformed_test_file_path=os.path.join(artifact_dir,'test.npy')
    transformed_object_file_path=os.path.join(artifact_dir,'preprocessor.pkl')

class DataTransformation:
    def __init__(self,feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config=DataTransformationConfig()
        self.utils =MainUtils()
        
    @staticmethod
    def get_data(feature_store_file_path : str)-> pd.DataFrame:
        
        try:
            data=pd.read_csv(feature_store_file_path)
            data.rename(columns={'Good/Bad':TARGET_COLUMN},inplace=True)
            
            return data
        except Exception as e:
            raise Custom_exception(e,sys)
        
    def get_data_transformer_object(self):
        try:
            imputer_step=('imputer',SimpleImputer(strategy='constant',fil_value=0))
            
            scaler_step=('scaling',RobustScaler())
            
            preprocessor=Pipeline(steps=[
                imputer_step,scaler_step
            ])
            
            return preprocessor
        except Exception as e:
            raise Custom_exception(e,sys)
        
        
    def initiate_data_transformation(self):
        logging.info('entered initiate data transformation method of data transformation class')
        
        try:
            dataframe=self.get_data(feature_store_file_path=self.feature_store_file_path)
            
            x=dataframe.drop(TARGET_COLUMN,axis=1)
            #y=dataframe[TARGET_COLUMN]
            
            y=np.where(dataframe[TARGET_COLUMN]==-1,0,1)
            
            xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=1,test_size=0.2)
            
            preprocessor=self.get_data_transformer_object()
            x_train_scaled=preprocessor.fit_transform(xtrain)
            x_test_scaled=preprocessor.transform(xtest)
            
            preprocessor_path =self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path),exist_ok =True)
            
            self.utils.save_object(file_path =preprocessor_path,obj=preprocessor)
            
            train_arr =np.c[x_train_scaled,np.arr(ytrain)]
            test_arr =np.c[x_test_scaled,np.arr(ytest)]
            
            return (train_arr,test_arr,preprocessor_path)
        
        except Exception as e:
            raise Custom_exception(e,sys)
            
                
            
            
             
            
            
            
            
        
        
         
        