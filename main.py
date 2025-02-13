from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipelines.Predict_pipeline import CustomData, PredictPipeline

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    tgt_column_name = 'math_score'
    r_columns = ['math_score']

    #dynamic categorization
    train_df = pd.read_csv(train_data)
    num_columns = train_df.select_dtypes(exclude="object").columns.tolist()
    cat_columns = train_df.select_dtypes(include="object").columns.tolist()
    num_columns.remove(tgt_column_name)

    data_transformation = DataTransformation()
    data_transformation.get_data_transformation_object(numerical_columns=num_columns, categorical_columns=cat_columns)
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data, 
        numerical_columns=num_columns, categorical_columns=cat_columns,
        target_column_name=tgt_column_name, removed_columns=r_columns
        )

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)

    # Make a prediction on unseen data
    data = CustomData(
        gender='male',
        race_ethnicity='group C',
        parental_level_of_education='associate\'s degree',
        lunch='standard',
        test_preparation_course='completed',
        reading_score=90,
        writing_score=85

    )
  
    predict_df = data.get_data_as_dataframe()
    print(predict_df)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(predict_df)
    print("Predicted math score: {:.4f}".format(results[0]))