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

            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
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
