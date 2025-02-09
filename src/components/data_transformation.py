import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.data_categories import (
    numerical_columns, 
    categorical_columns, 
    target_column_name
    )
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pk1')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        This function is responsible for the transformation of the numerical and categorical values in the dataset
        '''
        try:
            '''
            numerical_columns = ['reading_score','writing_score','math_score']

            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            '''
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical values standard scaling completed')

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Categorical values one hot encoding and standard scaling completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Train and test datasets loaded')

            logging.info('Initialize preprocessor object')
            preprocessor_obj = self.get_data_transformation_object()


            input_train_features_df = train_df.drop(columns=[target_column_name], axis=1)
            target_train_feature_df = train_df[target_column_name]

            input_test_features_df = test_df.drop(columns=[target_column_name], axis=1)
            target_test_feature_df = test_df[target_column_name]

            logging.info(
                f'Applying preprocessing to train and test dataframes using preprocessor object'
            )
    
            input_feature_train_array = preprocessor_obj.fit_transform(input_train_features_df)
            input_feature_test_array = preprocessor_obj.transform(input_test_features_df)

            train_array = np.c_[
                input_feature_train_array, np.array(target_train_feature_df)
            ]
            test_array = np.c_[
                input_feature_test_array, np.array(target_test_feature_df)
            ]

            logging.info('Saving preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
