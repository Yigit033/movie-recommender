import pandas as pd
import numpy as np
import ast
import seaborn as sns
import warnings
from config.config import PALETTE

class DataPreprocessor:
    
    def __init__(self):
        # Set visualization styles
        sns.set_theme(context='notebook', palette=PALETTE, style='white')
        warnings.simplefilter(action='ignore', category=FutureWarning)
    
    def preprocess_movies(self, movies):
        """Preprocess movies dataframe"""
        movies = movies.copy()
        movies["id"] = pd.to_numeric(movies["id"], errors='coerce')
        movies.dropna(subset=["id"], inplace=True)
        return movies
    
    def preprocess_and_merge(self, movies, ratings):
        """Merge and preprocess movies and ratings data"""
        # Initial preprocessing
        movies = self.preprocess_movies(movies)
        
        # Merge datasets
        merged_df = pd.merge(ratings, movies, left_on="movieId", right_on="id", how="inner")
        
        # Create features
        merged_df = self._create_features(merged_df)
        
        # Process genres
        merged_df = self._process_genres(merged_df)
        
        return merged_df
    
    def _create_features(self, df):
        """Create additional features"""
        df = df.copy()
        df["is_in_collection"] = df["belongs_to_collection"].apply(
            lambda x: 0 if pd.isnull(x) else 1
        )
        df = df[df["status"] == "Released"]
        df["releated_year"] = pd.to_datetime(df["release_date"]).dt.year
        return df
    
    def _process_genres(self, df):
        """Process genre information"""
        df = df.copy()
        
        # Convert genres string to list
        df["genres"] = df["genres"].apply(
            lambda x: ast.literal_eval(x) if not pd.isnull(x) else []
        )
        
        # Extract genre names
        df["genre_names"] = df["genres"].apply(
            lambda x: [genre["name"] for genre in x]
        )
        
        # Create genre dummies
        genre_dummies = df["genre_names"].str.join("|").str.get_dummies()
        df = pd.concat([df, genre_dummies], axis=1)
        
        # Clean up
        df.drop(["genres", "genre_names"], axis=1, inplace=True)
        
        return df