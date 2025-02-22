import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from config.config import EMBEDDING_DIM, EPOCHS, BATCH_SIZE, TEST_SIZE, RANDOM_STATE

class MovieRecommender:
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.model = None


    def prepare_data(self, merged_df):
        """Prepare data for the recommender model"""
        # Convert data types
        merged_df = merged_df.copy()
        merged_df["userId"] = merged_df["userId"].astype("int")
        merged_df["movieId"] = merged_df["movieId"].astype("int")
        merged_df["rating"] = merged_df["rating"] / merged_df["rating"].max()
        
        # Split data
        train, test = train_test_split(
            merged_df, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE
        )
        
        return self._extract_features(train, test)
    
    def _extract_features(self, train, test):
        """Extract features from train and test sets"""
        train_data = {
            'user_ids': train['userId'].values,
            'movie_ids': train['movieId'].values,
            'ratings': train['rating'].values
        }
        
        test_data = {
            'user_ids': test['userId'].values,
            'movie_ids': test['movieId'].values,
            'ratings': test['rating'].values
        }
        
        num_users = max(train_data['user_ids'].max(), test_data['user_ids'].max()) + 1
        num_movies = max(train_data['movie_ids'].max(), test_data['movie_ids'].max()) + 1
        
        return train_data, test_data, num_users, num_movies
    
    def build_model(self, num_users, num_movies):
        """Build the neural network model"""
        user_input = layers.Input(shape=(1,), name='user_input')
        movie_input = layers.Input(shape=(1,), name='movie_input')
        
        user_embedding = layers.Embedding(
            input_dim=num_users, 
            output_dim=self.embedding_dim
        )(user_input)
        user_embedding = layers.Flatten()(user_embedding)
        
        movie_embedding = layers.Embedding(
            input_dim=num_movies, 
            output_dim=self.embedding_dim
        )(movie_input)
        movie_embedding = layers.Flatten()(movie_embedding)
        
        concatenated = layers.Concatenate()([user_embedding, movie_embedding])
        dense_1 = layers.Dense(128, activation='relu')(concatenated)
        dense_2 = layers.Dense(64, activation='relu')(dense_1)
        output = layers.Dense(1)(dense_2)
        
        self.model = models.Model(inputs=[user_input, movie_input], outputs=output)
        self.model.compile(optimizer='adam', loss='mse')
        
        return self.model
    
    def train(self, train_data, test_data):
        """Train the model"""
        history = self.model.fit(
            [train_data['user_ids'], train_data['movie_ids']],
            train_data['ratings'],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(
                [test_data['user_ids'], test_data['movie_ids']], 
                test_data['ratings']
            ),
            verbose=1
        )
        return history        

    def get_recommendations(self, user_id, merged_df, movies_df, top_n=10):
        """Get movie recommendations for a specific user"""
        try:
            # Clean and convert movie IDs to numeric
            movies_df = movies_df.copy()
            movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
            movies_df = movies_df.dropna(subset=['id'])
            movies_df['id'] = movies_df['id'].astype('int64')
            
            # Get rated movies
            rated_movies = merged_df[merged_df['userId'] == user_id][['movieId', 'rating', 'title']]
            
            # Find unrated movies
            all_movie_ids = merged_df['movieId'].unique()
            unseen_movie_ids = np.setdiff1d(all_movie_ids, rated_movies['movieId'])
            
            # Prepare prediction inputs
            unseen_movie_ids_input = np.array(unseen_movie_ids)
            user_input_array = np.full(len(unseen_movie_ids_input), user_id)
            
            # Get predictions
            predicted_ratings = self.model.predict([user_input_array, unseen_movie_ids_input])
            
            # Create recommendations DataFrame
            recommendations = pd.DataFrame({
                'movieId': unseen_movie_ids_input,
                'predicted_rating': predicted_ratings.flatten()
            })
            
            # Convert movieId to int64
            recommendations['movieId'] = recommendations['movieId'].astype('int64')
            
            # Get top recommendations
            top_recommendations = recommendations.sort_values(
                by='predicted_rating', 
                ascending=False
            ).head(top_n)
            
            # Add movie titles
            top_recommendations = pd.merge(
                top_recommendations, 
                movies_df[['id', 'title']], 
                left_on='movieId', 
                right_on='id',
                how='left'
            )
            
            return rated_movies, top_recommendations[['title', 'predicted_rating']]
            
        except Exception as e:
            print(f"Error in get_recommendations: {e}")
            raise