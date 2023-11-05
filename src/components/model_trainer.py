import os
import sys
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from catboost import  CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Starting Model Training')
            logging.info('Splitting training and test input data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models={
                "RandomForest": RandomForestClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
               # "Support Vector Machine":SVC(),
                "XGBoost":XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoostClassifier":AdaBoostClassifier(),
               # "KNNClassifier":KNeighborsClassifier()

            }

            params={
                "DecisionTree": {
                    'criterion':['gini', 'entropy', 'log_loss'],
                    'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
               
                # "KNNClassifier":{
                #     'n_neighbors':[5,7,9,11],
                #     # 'weights':['uniform','distance'],
                #     # 'algorithm':['ball_tree','kd_tree','brute']
                # },
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Classifier":{
                    'depth': [6,8,10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostClassifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                # "Support Vector Machine":{
                #     'C': [0.1,1, 10, 100], 
                #     'gamma': [1,0.1,0.01,0.001],
                #     'kernel': ['rbf', 'poly', 'sigmoid']

                # }
                
            }
            logging.info("Training Model")
            model_report:dict=evaluate_models(X=X_train,y=y_train,x_test=X_test,y_test=y_test ,models=models, param=params)
            print(model_report)
            logging.info("Predicting outputs")

            ##Get the best model score from disctionary
            
            best_model_score=max(sorted(model_report.values()))

            logging.info("Evaluated Model")
            ##Get the best model name from dictionary


            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            

            logging.info("Saving Model pickle file")
            save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
            )
            logging.info("Saved model Pickle file")
            predicted=best_model.predict(X_test)
            logging.info("Best model Found on train and tess data")
            r2_score_val= r2_score(y_test,predicted)
            return r2_score_val
        

        except Exception as e:
            raise CustomException(e,sys)



    

