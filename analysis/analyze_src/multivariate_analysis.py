from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract base class for multivariate analysis strategies
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame) -> None:
        """Perform multivariate analysis by generating a correlation heatmap and a pairplot.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
        """
        self.generate_correlation_heatmap(df)  # Generate the correlation heatmap
        self.generate_pairplot(df)  # Generate the pairplot

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Abstract method to generate a correlation heatmap.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame) -> None:
        """Abstract method to generate a pairplot for the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
        """
        pass


# Concrete implementation of multivariate analysis
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Generate and display a correlation heatmap for numerical features.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
        """
        plt.figure(figsize=(12, 10))  # Set the figure size
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)  # Plot heatmap
        plt.title("Correlation Heatmap")  # Add a title to the plot
        plt.show()  # Display the plot

    def generate_pairplot(self, df: pd.DataFrame) -> None:
        """Generate and display a pairplot for the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
        """
        sns.pairplot(df)  # Create pairplot for all features in the DataFrame
        plt.suptitle("Pair Plot of Selected Features", y=1.02)  # Add a title to the pairplot
        plt.show()  # Display the plot
