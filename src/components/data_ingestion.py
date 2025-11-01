import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split  # type: ignore

sys.path.append(str(Path(__file__).parent.parent))

from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.exception import CustomException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion method or component")
        try:
            df: DataFrame = pd.read_csv("notebooks/data/stud.csv")
            logger.info("Read the dataset as dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True,
            )
            df.to_csv(
                self.ingestion_config.raw_data_path, index=False, header=True
            )

            logger.info("Train test split initiated")

            train_set: DataFrame
            test_set: DataFrame
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True,
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True,
            )

            logger.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as error:
            raise CustomException(error, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
