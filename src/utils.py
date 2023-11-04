import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X,y,x_test,y_test,models):
    try:
        report={}
       # print(models)
        for i in range(len(list(models))) :
            model=list(models.values())[i]
           # print(model)
            model.fit(X,y)

            y_train_pred=model.predict(X)
           # print(i, "Has this Ytrain", y_train_pred)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y,y_train_pred)
          #  print(train_model_score)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            return report


    except Exception as e:
        raise CustomException(e,sys)