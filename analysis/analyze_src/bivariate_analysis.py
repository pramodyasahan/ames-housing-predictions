from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract base class for bivariate analysis strategies
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """Abstract method to perform bivariate analysis on two features.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature1 (str): The first feature/column to analyze.
            feature2 (str): The second feature/column to analyze.
        """
        pass


# Concrete strategy for analyzing the relationship between two numerical features
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """Perform bivariate analysis on two numerical features using a scatter plot.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature1 (str): The first numerical feature.
            feature2 (str): The second numerical feature.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(f"{feature1}")
        plt.ylabel(f"{feature2}")
        plt.show()


# Concrete strategy for analyzing the relationship between a categorical and a numerical feature
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """Perform bivariate analysis on a categorical and a numerical feature using a box plot.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature1 (str): The categorical feature.
            feature2 (str): The numerical feature.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(f"{feature1}")
        plt.ylabel(f"{feature2}")
        plt.xticks(rotation=45)
        plt.show()


# Context class to execute the chosen bivariate analysis strategy
class BivariateAnalysis:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """Initialize the analyzer with a specific bivariate analysis strategy.

        Args:
            strategy (BivariateAnalysisStrategy): The analysis strategy to use.
        """
        self.strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """Execute the analysis using the selected strategy.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature1 (str): The first feature/column to analyze.
            feature2 (str): The second feature/column to analyze.
        """
        self.strategy.analyze(df, feature1, feature2)
