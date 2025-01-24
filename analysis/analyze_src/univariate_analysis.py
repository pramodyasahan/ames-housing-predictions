from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract base class for univariate analysis strategies
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """Abstract method to perform univariate analysis on a feature.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature (str): The feature/column to analyze.
        """
        pass


# Concrete strategy for numerical feature analysis
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """Perform univariate analysis on a numerical feature by plotting its distribution.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature (str): The numerical feature/column to analyze.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=40)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


# Concrete strategy for categorical feature analysis
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """Perform univariate analysis on a categorical feature by plotting its distribution.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature (str): The categorical feature/column to analyze.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()

    # Context class to execute the chosen univariate analysis strategy


class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy) -> None:
        """Initialize the analyzer with a specific univariate analysis strategy.

        Args:
            strategy (UnivariateAnalysisStrategy): The analysis strategy to use.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """Execute the analysis using the selected strategy.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature (str): The feature/column to analyze.
        """
        self._strategy.analyze(df, feature)
