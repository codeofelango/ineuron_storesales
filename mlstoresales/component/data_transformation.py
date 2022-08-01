from cgi import test
from sklearn import preprocessing
from mlstoresales.exception import HousingException
from mlstoresales.logger import logging
from mlstoresales.entity.config_entity import DataTransformationConfig 
from mlstoresales.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
import sys,os
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
from mlstoresales.constant import *
from mlstoresales.util.util import read_yaml_file,save_object,save_numpy_array_data,data_transformation_load_data,load_data
from kneed import KneeLocator
from sklearn.cluster import KMeans


#   longitude: float
#   latitude: float
#   housing_median_age: float
#   total_rooms: float
#   total_bedrooms: float
#   population: float
#   households: float
#   median_income: float
#   median_house_value: float
#   ocean_proximity: category
#   income_cat: float


class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, current_year=2022):
        try:
            self.current_year = current_year
            self.cluster = None
            if self.current_year is None:
                raise HousingException("Year Mube be assigned")

        except Exception as e:
            raise HousingException(e, sys) from e

    def fit(self, X, y=None):
        wcss=[]
        for i in range(1,11):
            kmeans=KMeans(n_clusters=i, init='k-means++',random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_) 
    
        kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
        total_clusters=kn.knee
        self.cluster = KMeans(n_clusters=total_clusters, init='k-means++',random_state=42)
        self.cluster.fit(X)
        return self

    def transform(self, X, y=None):
        try:
            logging.info("Transforming data")
            data = X.copy()
            generated = pd.DataFrame(self.cluster.predict(data) , columns=['cluster'])
            

            return generated
        except Exception as e:
            raise HousingException(e, sys) from e





class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise HousingException(e,sys) from e

    

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)
            columns_to_cluster = dataset_schema[COLUMNS_CLUSTER_KEY]

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]

            feature_pipeline = Pipeline( steps = [
                    ('imputer' , SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('feature_generator', FeatureGenerator()),
            ])
            num_pipeline = Pipeline(steps=[
                                    ('imputer', SimpleImputer(strategy="median")),
                                    ('scaler', StandardScaler())
                                    ]
                            )

            cat_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy="most_frequent")),
            ('one_hot_encoder', OneHotEncoder(sparse=False, handle_unknown='ignore')),
            ('scaler', StandardScaler(with_mean=False))
            ]
            )
            

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessing = ColumnTransformer([
                    ('feature_generator', feature_pipeline, columns_to_cluster),
            ('num_pipeline', num_pipeline, numerical_columns),
            ('cat_pipeline', cat_pipeline, categorical_columns),
            ])
            return preprocessing

        except Exception as e:
            raise HousingException(e,sys) from e  

    def data_transformation_load_data(self, file_path, selected_columns) -> pd.DataFrame:
        try:
            load_df = pd.read_csv(file_path, usecols=selected_columns)
            load_df["Item_Fat_Content"] = load_df["Item_Fat_Content"].map(
                {"Low Fat": 'Low Fat', "LF": "Low Fat", 'low fat': "Low Fat", "Regular": "Regular"})
            load_df["Item_Identifier"] = load_df["Item_Identifier"].apply(lambda x: x[0:2]).map(
                {"FD": "Food", "DR": "Drink", "NC": "Non_consumable"})
            
            return load_df
        except Exception as e:
            raise HousingException(e, sys) from e       

    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()


            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            # columns_to_load  = dataset_schema[COLUMNS_KEY].keys()
            # logging.info(f"Loading training and test data as pandas dataframe.")
            # train_df = data_transformation_load_data(file_path=train_file_path, selected_columns=columns_to_load)
            
            # test_df = data_transformation_load_data(file_path=test_file_path, selected_columns=columns_to_load)

            schema = read_yaml_file(file_path=schema_file_path)


            columns_to_load  = schema[COLUMNS_KEY].keys()
           
            logging.info(f"Selected columns: {columns_to_load}")

            logging.info(f"Loading training and test data as pandas dataframe.")
            # train_df = self.data_transformation_load_data(file_path=train_file_path, selected_columns=columns_to_load)

            # test_df = self.data_transformation_load_data(file_path=test_file_path, selected_columns=columns_to_load)

            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)
            target_column_name = schema[TARGET_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            print(f"before input_feature_train_df{input_feature_train_df}")
            print(f"before input_feature_test_df {input_feature_test_df}")

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            print(f"after input_feature_train_df {input_feature_train_df}")
            print(f"after input_feature_test_df {input_feature_test_df}")

            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")