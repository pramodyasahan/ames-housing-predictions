from abc import ABC, abstractmethod
import pandas as pd


# Define an abstract base class for data inspection strategies
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """Abstract method to inspect the DataFrame.
        This method must be implemented by any concrete inspection strategy.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.
        """
        pass


# Concrete implementation of the strategy to inspect data types and non-null counts
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """Prints data types and non-null counts for the DataFrame columns.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.
        """
        print("\n Data Type and Non-null Counts: ")
        print(df.info())


# Concrete implementation of the strategy to inspect summary statistics
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """Prints summary statistics for numerical and categorical columns.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.
        """
        print("\n Summary Statistics: (Numerical Features)")
        print(df.describe())  # Summary statistics for numerical features

        print("\n Summary Statistics: (Categorical Features)")
        print(df.describe(include=["O"]))  # Summary statistics for categorical features


# Context class to execute a specific data inspection strategy
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """Initializes the DataInspector with a specific inspection strategy.

        Args:
            strategy (DataInspectionStrategy): The inspection strategy to use.
        """
        self._strategy = strategy

    def execute(self, df: pd.DataFrame):
        """Executes the selected inspection strategy on the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to inspect.
        """
        self._strategy.inspect(df)
