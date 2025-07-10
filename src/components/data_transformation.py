import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessing pipeline for numerical and categorical features.
        """
        try:
            # ✅ 'math_score' is the target column, so exclude it
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            # ✅ Fix: with_mean=False to avoid sparse matrix error
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("onehotencoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Numerical and categorical pipelines created successfully.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns),
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Transforms the training and testing datasets and saves the preprocessor object.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully.")
            logging.info(f"Train Columns: {train_df.columns.tolist()}")
            logging.info(f"Test Columns: {test_df.columns.tolist()}")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = 'math_score'

            input_features_train = train_df.drop(columns=[target_column], axis=1)
            target_feature_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column], axis=1)
            target_feature_test = test_df[target_column]

            logging.info("Input features (train): %s", input_features_train.columns.tolist())

            logging.info("Starting data transformation...")

            input_features_train_transformed = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_transformed = preprocessing_obj.transform(input_features_test)

            logging.info("Data transformation completed.")

            train_arr = np.c_[input_features_train_transformed, np.array(target_feature_train)]
            test_arr = np.c_[input_features_test_transformed, np.array(target_feature_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
