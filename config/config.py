# Configuration settings for the movie recommender system

# Data paths
MOVIES_PATH = 'datasets/movies_metadata.csv'
RATINGS_PATH = 'datasets/ratings_small.csv'

# Model parameters
EMBEDDING_DIM = 50
EPOCHS = 1
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Visualization settings
PALETTE = ['#f28a30', '#3b94d9', '#e0c72c', '#6a4d8e', '#e63946', '#2a9d8f', '#264653']
HEATMAP_COLORS = ['#8B0000', '#f0e68c', '#0e560c']

# Feature lists
NUMERIC_FEATURES = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']