import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd


# Define an abstract class for data ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from zip file"""
        pass


# Implement a concrete class for ZIP ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Ingest data from zip file"""
        # Ensure the file is a zip file
        if not file_path.endswith(".zip"):
            raise ValueError("File extension must be .zip")

        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_file:
            zip_file.extractall("extracted_data")

        # Find the extracted CSV file
        extracted_data = os.listdir("extracted_data")
        csv_file = [f for f in extracted_data if f.endswith(".csv")]

        if len(csv_file) == 0:
            raise FileNotFoundError("No .csv files found in extracted_data")
        if len(csv_file) > 1:
            raise ValueError("More than one .csv files found in extracted_data")

        # Read the csv in to dataframe
        csv_file_path = os.path.join("extracted_data", csv_file[0])
        df = pd.read_csv(csv_file_path)

        return df


class DataIngestorFactory:
    @staticmethod
    def ingest(file_extension: str) -> DataIngestor:
        """Return the appropriate data ingestor"""
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")
