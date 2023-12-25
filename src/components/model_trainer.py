import pandas as pd
import os
import sys
import numpy as np

from src.utils import save_object,evaluate_models
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge,GammaRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "RandomForest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "SVR": SVR(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Ridge" : Ridge(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "GammaRegressor": GammaRegressor()
            }
            
            params = {
                "RandomForestRegressor":{
                    'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    'max_features':['sqrt','log2'],
                    'max_depth':[23,57,87,93,20]
                },
                "LinearRegression":{
                    
                },
                "SVR":{
                    'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    'epsilon':[0.1,0.01,0.001,0.0001],
                    'degree':[1,2,3,4,5],
                    'C': [6,8,10,12,14]
                    
                },
                "DecisionTreeRegresor":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best', 'random'],
                    'max_features':['auto', 'sqrt', 'log2'],
                    'max_depth':[100,432,550,212]
                    
                },
                "XGBRegressor":{
                    
                },
                "Ridge": {
                    'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                    'max_iter':[20,300,500,623,723],
                    'alpha':[1,2,3,4,5]
                },
                "AdaBoostRegressor":{
                    'n_estimators':[23,56,78,93],
                    'loss':['linear', 'square', 'exponential'],
                    'learning_rate':[1,0.1,0.02,0.001]
                    
                },
                "GammaRegressor":{
                    'solver':['lbfgs', 'newton-cholesky'],
                    'max_iter':[122,322,453,212,67],
                    'tol':[0.001,0.01,0.0001,0.002]
                }
            }
            
            
            
            model_report:dict= evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                               models= models,param=params)
            
            best_model_score = max(sorted(model_report.values()))
            
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing dataset")
            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj =best_model
            )
            
            predicted =best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            
            return r2_square
        
        
        
        
        except Exception as e:
            raise CustomException(e,sys)
