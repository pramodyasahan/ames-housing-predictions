from abc import ABC, abstractmethod

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Define an abstract base class for missing value analysis
class MissingValueAnalysis(ABC):
    def analyze(self, df: pd.DataFrame) -> None:
        """Perform missing value analysis by identifying and visualizing missing data.

        Args:
            df (pd.DataFrame): The DataFrame to analyze for missing values.
        """
        self.identify_missing_values(df)  # Identify missing values
        self.visualize_missing_values(df)  # Visualize missing values

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        """Abstract method to identify missing values in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze for missing values.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        """Abstract method to visualize missing values in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze for missing values.
        """
        pass


# Concrete implementation of missing value analysis
class SimpleMissingValueAnalysis(MissingValueAnalysis):
    def identify_missing_values(self, df: pd.DataFrame) -> None:
        """Identify and print missing values count for each column in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze for missing values.
        """
        print("\nMissing values count by Column")
        missing_values = df.isnull().sum()  # Calculate missing values per column
        print(missing_values[missing_values > 0])  # Print columns with missing values

    def visualize_missing_values(self, df: pd.DataFrame) -> None:
        """Visualize missing values in the DataFrame using a heatmap.

        Args:
            df (pd.DataFrame): The DataFrame to analyze for missing values.
        """
        print("\nVisualize missing values")
        plt.figure(figsize=(12, 8))  # Set figure size for better visualization
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")  # Plot heatmap of missing values
        plt.title("Missing values")  # Add a title to the heatmap
        plt.show()
