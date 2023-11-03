from typing import Any
from src.exception import CustomException
from src.logger import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from src.utils import save_object

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifact", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['Longitude','Latitude','NO2 Hour','NO2 Value','Month','day']
            categorical_features = ['Station','NO2 Quality']
   
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("Scalar", StandardScaler(with_mean=False)),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder", OneHotEncoder()),
                    ("StandardScalar", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Numerical columns Standard Scaling completed.")

            logging.info("Catergorical Column Categorical encoding completed.")

            preprocessor = ColumnTransformer(
                [
                    ("Numerical_pipeline", numerical_pipeline, numerical_features),
                    ("Categorical_pipeline",categorical_pipeline,categorical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("tead train and test data completed")

            logging.info("Obtaining preprocessing object...") 
            preprocessing_object =self.get_data_transformer_object()

            target_colum_name='AirQuality'
      
            input_feature_train_df=train_df.drop(columns=target_colum_name, axis=1)
            target_feature_train_df=train_df[target_colum_name]
           
            input_feature_test_df=test_df.drop(columns=target_colum_name,axis=1)

            target_feature_test_df=test_df[target_colum_name]
   
        

            logging.info("Applying preprocessor obj on train and test df")
         
            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)
 
          
            logging.info(f"Saving preprocessing object.")
            
          

        
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

           
            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_object

            )

            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_ob_file_path,
            )




        except Exception as e:
            raise CustomException(e,sys)
 