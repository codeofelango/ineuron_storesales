import os
from re import S
import sys
from mlstoresales.logger import logging
from mlstoresales.exception import HousingException
from mlstoresales.util.util import load_object
import numpy as np

import pandas as pd


class Prediction_Data:

    def __init__(self, Item_Identifier: str,
                 Item_Fat_Content: str,
                 Item_Type: str,
                 Outlet_Identifier: str,
                 Outlet_Type: str,
                 Item_MRP: float,
                 Item_Visibility: float,
                 Item_Weight: float,
                 Outlet_Establishment_Year: int,
                 Outlet_Location_Type :str ,
                Outlet_Size: str,
                 Item_Outlet_Sales: float = None):

        try:
            self.Item_Identifier = Item_Identifier
            self.Item_Fat_Content = Item_Fat_Content
            self.Item_Type = Item_Type
            self.Outlet_Identifier = Outlet_Identifier
            self.Outlet_Type = Outlet_Type
            self.Item_MRP = Item_MRP
            self.Item_Visibility = Item_Visibility
            self.Item_Weight = Item_Weight
            self.Outlet_Establishment_Year = Outlet_Establishment_Year
            self.Outlet_Location_Type = Outlet_Location_Type
            self.Outlet_Size = Outlet_Size
            self.Item_Outlet_Sales = Item_Outlet_Sales
        except Exception as e:
            raise HousingException(e, sys) from e

    def get_housing_input_data_frame(self):

        try:
            housing_input_dict = self.get_housing_data_as_dict()
            return pd.DataFrame(housing_input_dict)
        except Exception as e:
            raise HousingException(e, sys) from e

    def get_housing_data_as_dict(self):
        try:
            input_data = {
                "Item_Identifier": [self.Item_Identifier],
                "Item_Fat_Content": [self.Item_Fat_Content],
                "Item_Type": [self.Item_Type],
                "Outlet_Identifier": [self.Outlet_Identifier],
                "Outlet_Type": [self.Outlet_Type],
                "Item_MRP": [self.Item_MRP],
                "Item_Visibility": [self.Item_Visibility],
                "Item_Weight": [self.Item_Weight],
                'Outlet_Location_Type': [self.Outlet_Location_Type],
                'Outlet_Size': [self.Outlet_Size],
                "Outlet_Establishment_Year": [self.Outlet_Establishment_Year]
            }

            return input_data
        except Exception as e:
            raise HousingException(e, sys)


class App_predictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise HousingException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise HousingException(e, sys) from e

    def predict(self, X):
        try:
            model_path = os.path.join(self.model_dir , "model.pkl")
            model = load_object(file_path=model_path)
            logging.info(f"Best model: {model}")
            median_house_value = model.predict(X)
            return median_house_value
        except Exception as e:
            raise HousingException(e, sys) from e

    def predictwithtransform(self, X):
        try:
            preprocessed_path = os.path.join(self.model_dir , "preprocessed.pkl")
            logging.info(f"preprocessed_path: {preprocessed_path}")

            preprocessed_file = load_object(file_path=preprocessed_path)
            logging.info(f"preprocessed_file: {preprocessed_file}")
            logging.info(f"x from form: {X}")

#   "Item_Identifier": [self.Item_Identifier],
#                 "Item_Fat_Content": [self.Item_Fat_Content],
#                 "Item_Type": [self.Item_Type],
#                 "Outlet_Identifier": [self.Outlet_Identifier],
#                 "Outlet_Type": [self.Outlet_Type],
#                 "Item_MRP": [self.Item_MRP],
#                 "Item_Visibility": [self.Item_Visibility],
#                 "Item_Weight": [self.Item_Weight],
#                 'Outlet_Location_Type': [self.Outlet_Location_Type],
#                 'Outlet_Size': [self.Outlet_Size],
#                 "Outlet_Establishment_Year": [self.Outlet_Establishment_Year]


            preditvalue = [X.Item_Identifier,X.Item_Fat_Content,X.Item_Type,
            X.Outlet_Identifier,X.Outlet_Type,X.Item_MRP,X.Item_Visibility,X.Item_Weight,X.Outlet_Location_Type,
            X.Outlet_Size,X.Outlet_Establishment_Year]
            # preditvalue = [[1,2,1,
            # 1,2,2323,54354,767665,2,
            # 6,2000]]
            # inputs= np.array(preditvalue ).reshape(1,-1)

            # transformeddata= preprocessed_file.transform(preditvalue)
            # logging.info(f"transformeddata: {transformeddata}")
            # print(transformeddata)
            # print(len(transformeddata[0]))
            #Lets apply prediction

            # model_path = os.path.join(self.model_dir , "model.pkl")
            # logging.info(f"model_path: {model_path}")
            # model_file = load_object(file_path=model_path)
            # logging.info(f"model_file: {model_file}")
            # print(f"going to predict:{transformeddata}")
            # prediction = model_file.predict(transformeddata)
            # logging.info(f"prediction: {prediction}")
            # print(prediction)    
            # output = round(prediction[0], 2)
            output=65743
            print(output)         
            return output
        except Exception as e:
            raise HousingException(e, sys) from e