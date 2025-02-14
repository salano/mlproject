import os
import sys 
import dill

import numpy as np 
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
  
    
def evaluate_model(x_train, y_train, x_test, y_test, models, param):
    try:
        report: dict = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            logging.info("Train model [{}]".format(list(models.keys())[i]))
            print(list(models.keys())[i],'\n')
            #Hyperparameter
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=5)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            #end Hyperparameter
            model.fit(x_train, y_train)

            y_train_predict = model.predict(x_train)
            y_test_predict = model.predict(x_test)

            '''
            coefficient of determination - explains how well the independent variables
            explain the variability of the independent variable.
            High R2 suggests dependednt variable is hightly correlated with independent variables (Multicollinearity)
            '''

            train_model_r2_score = r2_score(y_train, y_train_predict)
            test_model_r2_score = r2_score(y_test, y_test_predict)

            train_model_mae_score = mean_absolute_error(y_train, y_train_predict)
            test_model_mae_score = mean_absolute_error(y_test, y_test_predict)

            train_model_mse_score = mean_squared_error(y_train, y_train_predict)
            test_model_mse_score = mean_squared_error(y_test, y_test_predict)

            train_model_rmse_score = np.sqrt(mean_squared_error(y_train, y_train_predict))
            test_model_rmse_score = np.sqrt(mean_squared_error(y_test, y_test_predict))

            print('Model performance for Training set')
            print("- Root Mean Squared Error: {:.4f}".format(train_model_rmse_score))
            print("- Mean Absolute Error: {:.4f}".format(train_model_mae_score))
            print("- R2 Score: {:.4f}".format(train_model_r2_score))

            print('----------------------------------')
            
            print('Model performance for Test set')
            print("- Root Mean Squared Error: {:.4f}".format(test_model_rmse_score))
            print("- Mean Absolute Error: {:.4f}".format(test_model_mse_score))
            print("- R2 Score: {:.4f} \n".format(test_model_r2_score))


            report[list(models.keys())[i]] = test_model_r2_score

        print(pd.DataFrame(report.items(), columns=['Model Name', 'R2_Score']).sort_values(by=["R2_Score"],ascending=False))

        return report
            
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
