### Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LinearRegression 

### Metrics
from sklearn.metrics import accuracy_score, r2_score
from dataclasses import dataclass
import os, sys
from src.utils import evaluate_model, save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(),
                "SVM": SVC(),
                "Decision Tree": DecisionTreeClassifier(),
                "Naive Bayes": GaussianNB(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Linear Regression": LinearRegression()
            }

            params={
                "Random Forest":{
                    'criterion':['gini','entropy'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression":{
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                },
                "SVM":{
                    'C': [0.1, 1, 10, 100, 1000], 
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': [0.1, 1, 10, 100, 1000]
                    },
                "Decision Tree":{
                    'criterion':['gini','entropy'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2']
                },
                "Naive Bayes":{
                    'var_smoothing': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                },
                "K-Nearest Neighbors":{
                    'n_neighbors':[5,7,9,11],
                    'weights':['uniform','distance'],
                    'algorithm':['auto','ball_tree','kd_tree','brute']
                },
                "Linear Regression":{
                    'copy_X':[True, False],
                    'fit_intercept':[True, False],
                    'n_jobs':[1,2,4],
                    'positive':[True, False]
                }
            }

            model_report = evaluate_model(X_train, y_train, X_test, y_test, models,params = params)
            print("Model Report:", model_report)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No good model found (score < 0.6)", sys)

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
            logging.info(f"Saving model to {self.model_trainer_config.trained_model_file_path}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
