import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from config.config import PALETTE, HEATMAP_COLORS, NUMERIC_FEATURES

class DataVisualizer:
    def __init__(self):
        self.palette = PALETTE
        self.cmap = LinearSegmentedColormap.from_list("custom_cmap", HEATMAP_COLORS)
    
    def create_visualizations(self, merged_df):
        """Create all visualizations"""
        self.plot_genre_distribution(merged_df)
        self.plot_rating_distribution(merged_df)
        self.plot_correlation_heatmap(merged_df)
    
    def plot_genre_distribution(self, merged_df):
        """Plot distribution of movies across genres"""
        genre_dummies = merged_df.select_dtypes(include=['uint8']).columns
        genre_counts = merged_df[genre_dummies].sum().sort_values(ascending=False)
        
        plt.figure(figsize=(10,6))
        sns.barplot(x=genre_counts.values, y=genre_counts.index, palette=self.palette)
        plt.title("Number of Movies in Each Genre")
        plt.xlabel("Number of Movies")
        plt.ylabel("Genres")
        plt.show()
    
    def plot_rating_distribution(self, merged_df):
        """Plot distribution of user ratings"""
        plt.figure(figsize=(10,6))
        sns.histplot(merged_df["rating"], bins=10, kde=False, color=self.palette[3])
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.title("Distribution of User Ratings")
        plt.show()
    
    def plot_correlation_heatmap(self, merged_df):
        """Plot correlation heatmap of numeric features"""
        correlation_matrix = merged_df[NUMERIC_FEATURES].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap=self.cmap, vmin=-1, vmax=1)
        plt.title('Correlation Heatmap of Movie Attributes')
        plt.show()