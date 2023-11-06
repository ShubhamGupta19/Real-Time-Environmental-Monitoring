import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:

            model_path='artifact/model.pkl'
            preprocessor_path='artifact/preprocessor.pkl'
            model=load_object(filepath=model_path)
            preprocessor=load_object(filepath=preprocessor_path)
            scaled_features=preprocessor.transform(features)
            preds=model.predict(scaled_features)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(self,
        station: str,
        no2_hour: int,
        no2_quality: str,
        no2_value: int,
        month: int,
        day: int,
        longitude: float,  
        latitude: float):  

        self.station = station
        self.no2_hour = no2_hour
        self.no2_quality = no2_quality
        self.no2_value = no2_value
        self.month = month
        self.day = day
        self.longitude = longitude  
        self.latitude = latitude   

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Station": [self.station],
                "NO2 Hour": [self.no2_hour],
                "NO2 Quality": [self.no2_quality],
                "NO2 Value": [self.no2_value],
                "Month": [self.month],
                "day": [self.day],
                "Longitude": [self.longitude],  
                "Latitude": [self.latitude],    
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

