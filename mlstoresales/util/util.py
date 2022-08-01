import yaml
from mlstoresales.exception import HousingException
import os,sys
import numpy as np
import dill
import pandas as pd
from mlstoresales.constant import *


def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise HousingException(e,sys)


def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HousingException(e,sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise HousingException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise HousingException(e, sys) from e


def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise HousingException(e,sys) from e


def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise HousingException(e,sys) from e


def model_evaluation_load_data(file_path: str, selected_columns: str):
    load_df = pd.read_csv(file_path, usecols=selected_columns)
    load_df["Item_Fat_Content"] = load_df["Item_Fat_Content"].map(
        {"Low Fat": 'Low Fat', "LF": "Low Fat", 'low fat': "Low Fat", "Regular": "Regular"})
    return load_df
    

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
def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    try:
        datatset_schema = read_yaml_file(schema_file_path)

        schema = datatset_schema[DATASET_SCHEMA_COLUMNS_KEY]

        dataframe = pd.read_csv(file_path)

        error_messgae = ""

        # selected_columns = datatset_schema[COLUMNS_KEY].keys()

        # dataframe = pd.read_csv(file_path, usecols=selected_columns)
        # dataframe["Item_Fat_Content"] = dataframe["Item_Fat_Content"].map(
        # {"Low Fat": 'Low Fat', "LF": "Low Fat", 'low fat': "Low Fat", "Regular": "Regular"})

        print(dataframe["Item_Identifier"].unique())
        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_messgae = f"{error_messgae} \nColumn: [{column}] is not in the schema."
        if len(error_messgae) > 0:
            raise Exception(error_messgae)
        return dataframe

    except Exception as e:
        raise HousingException(e,sys) from e