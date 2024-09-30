import sys
from typing import Generator,List,Tuple
import os
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from src.constant import *
from src.utils.main_utils import MainUtils
from src.logger import logging
from src.exception import Custom_exception
from sklearn.model_selection import GridSearchCV,train_test_split

from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    artifact_folder =os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder,'model.pkl')
    excepted_accuracy=0.45
    model_config_file_path = os.path.join('config','model.yaml')
    
class ModelTrainer:
    
    def __init__(self):
        self.models={'xgboost':XGBClassifier(),
                     'svm':SVC(),'RandForst':RandomForestClassifier(),
                     'gradient':GradientBoostingClassifier(),
                     }
        
        self.model_trainer_config=ModelTrainingConfig()
        
        self.utils=MainUtils()
        
    def evaluate_models(self,x,y,models):
        
        try:
            xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=1,test_size=0.2)
        
            report={}
        
            for i in range(len(models)):
                model=list(model.values())[i]
            
                model.fit(xtrain,ytrain)
            
                ypredict=model.predict(ytest)
            
                score=accuracy_score(ytest,ypredict)
            
                report[list(models.keys())[i]]=score
            
            return report
        
        except Exception as e:
            raise Custom_exception(e,sys)
        
        
    def get_best_model(self,
                       xtrain:np.arr,
                       xtest:np.arr,
                       ytrain:np.arr,
                       ytest:np.arr):
        
        try:
            
            model_report:dict = self.evaluate_models(xtrain=xtrain,
                                                     ytrain=ytrain,
                                                     xtest=xtest,
                                                     ytest=ytest,
                                                     models=self.models)
            
            print(model_report)
            
            best_model_score = max(sorted(model_report.values()))
            
            value=list(model_report.values())
            key=list(model_report.keys())
            index=value.index(best_model_score)
            best_model_name=key[index]
            
            best_model_object = self.models[best_model_name]
            
            return best_model_name,best_model_object,best_model_score
        
        except Exception as e:
            raise Custom_exception(e,sys)
        
        
    def finetune_best_mode(self,best_model_name,best_model_object,best_model_score,xtrain,ytrain):
        try:
            model_param_grid =self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)['model_selection']['model'][best_model_name]['search_param_grid']
        
            grid_search=GridSearchCV(best_model_object,param_grid=model_param_grid,cv=5,verbose=1)
        
            grid_search.fit(xtrain,ytrain)
        
            best_params=grid_search.best_params_
        
            print('best_params is ',best_params)
        
            finetuned_model=best_model_object.set_params(**best_params)
        
            return finetuned_model
        
        except Exception as e:
            raise Custom_exception(e,sys)
        
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Splitting training and testing input and target feature")


            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )


           


            logging.info(f"Extracting model config file path")




           






            logging.info(f"Extracting model config file path")


            model_report: dict = self.evaluate_models(X=x_train, y=y_train, models=self.models)


            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))


            ## To get best model name from dict


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]




            best_model = self.models[best_model_name]




            best_model = self.finetune_best_model(
                best_model_name= best_model_name,
                best_model_object= best_model,
                X_train= x_train,
                y_train= y_train
            )


            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)
           
            print(f"best model name {best_model_name} and score: {best_model_score}")




            if best_model_score < 0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.6")
           
            logging.info(f"Best found model on both training and testing dataset")


 
       


            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_path}"
            )


            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)


            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
           
            return self.model_trainer_config.trained_model_path


           


           


        except Exception as e:
            raise Custom_exception(e, sys)

        
        
        
        
        
        
        
        
        
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    

    
    


