import pandas as pd
from config.config import MOVIES_PATH, RATINGS_PATH

class DataLoader:
    @staticmethod
    def load_data():
        """Load movie and ratings data from CSV files"""
        try:
            # Load movies data
            movies = pd.read_csv(MOVIES_PATH, low_memory=False)
            
            # Clean movie IDs
            movies["id"] = pd.to_numeric(movies["id"], errors='coerce')
            movies = movies.dropna(subset=["id"])
            movies["id"] = movies["id"].astype('int64')
            
            # Load ratings data
            ratings = pd.read_csv(RATINGS_PATH)
            ratings["movieId"] = ratings["movieId"].astype('int64')
            ratings["userId"] = ratings["userId"].astype('int64')
            
            return movies, ratings
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error while loading data: {e}")
            raise