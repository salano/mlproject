import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge, 
    ElasticNet, 
    Lasso
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from src.hyper_parameters import params


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pk1')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('splitting train and test datasets')
            X_train, Y_train, X_test, Y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
        
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(),
                'K-Neighbors Classifier': KNeighborsRegressor(),
                'XGBClassifier ': XGBRegressor(),
                'CatBoosting Classifier': CatBoostRegressor(verbose=False),
                'ADABoost Classifier': AdaBoostRegressor(),
                'Ridge Regression': Ridge(),
                'Lasso Regression' : Lasso(),
                'ElasticNet Regression': ElasticNet()
            }


            model_report: dict = evaluate_model(x_train=X_train,
                                                y_train=Y_train, x_test=X_test,
                                                y_test=Y_test, models=models,
                                                param=params)

            # Get best model score
            best_model_score = max(sorted(model_report.values()))

            #Get Best Model Name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No suitable model found')
            else:
                logging.info("Best model found on training dataset is [{}]".format(best_model))

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            pred_df=pd.DataFrame({'Actual Value': 
                                  Y_test, 'Predicted Value': predicted,
                                  'Difference': Y_test-predicted})
            print(pred_df)

            r2_square = r2_score(Y_test, predicted)

            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys)

