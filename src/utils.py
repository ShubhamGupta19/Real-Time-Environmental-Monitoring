import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X,y,x_test,y_test,models,param):
    try:
        report={}
       # print(models)
        for i in range(len(list(models))) :
            model=list(models.values())[i]
            para= param[list(models.keys())[i]]
            
            gs=GridSearchCV(estimator=model,param_grid=para,cv=3)
            gs.fit(X,y)

            model.set_params(**gs.best_params_)
            model.fit(X,y)


            print(model)
            y_train_pred=model.predict(X)
           # print(i, "Has this Ytrain", y_train_pred)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y,y_train_pred)
          #  print(train_model_score)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report


    except Exception as e:
        raise CustomException(e,sys)\


def load_object(filepath):
    try:
        with open(filepath,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)