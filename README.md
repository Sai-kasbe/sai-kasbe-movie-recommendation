Here is the complete content for a movie recommendation system implemented in Google Colab using Python. This includes installing necessary libraries, loading and preparing the data, training the model, evaluating it, and generating recommendations.

1. Install Required Libraries
python
Copy code
!pip install scikit-surprise
2. Import Libraries
python
Copy code
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
3. Download and Load the Data
python
Copy code
# Download the dataset
!wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip

# Load the ratings data
ratings = pd.read_csv('ml-latest-small/ratings.csv')
print("Ratings Data:")
print(ratings.head())
4. Prepare the Data
python
Copy code
# Prepare the data for the Surprise library
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
print("Data Prepared for Surprise Library")
5. Train-Test Split
python
Copy code
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
print("Train-Test Split Completed")
6. Train the Model
python
Copy code
algo = SVD()
algo.fit(trainset)
print("Model Training Completed")
7. Evaluate the Model
python
Copy code
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse:.4f}")
8. Make Recommendations
python
Copy code
def get_top_n_recommendations(predictions, user_id, n=10):
    # Get all predictions for the user
    user_predictions = [pred for pred in predictions if pred.uid == user_id]
    
    # Sort predictions by estimated rating
    user_predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get the top N movie IDs
    top_n_movie_ids = [pred.iid for pred in user_predictions[:n]]
    
    # Get movie titles
    movies = pd.read_csv('ml-latest-small/movies.csv')
    top_n_movies = movies[movies['movieId'].isin(top_n_movie_ids)]
    
    return top_n_movies[['movieId', 'title']]

# Get recommendations for a specific user
user_id = 1
top_n_movies = get_top_n_recommendations(predictions, user_id, n=10)
print("Top N Movie Recommendations for User ID 1:")
print(top_n_movies)
Full Program
Combining all the above steps, here is the full program:

python
Copy code
# Install the Surprise library
!pip install scikit-surprise

# Import necessary libraries
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Download and unzip the MovieLens dataset
!wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip

# Load the ratings data
ratings = pd.read_csv('ml-latest-small/ratings.csv')
print("Ratings Data:")
print(ratings.head())

# Prepare the data for the Surprise library
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
print("Data Prepared for Surprise Library")

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
print("Train-Test Split Completed")

# Train the SVD model
algo = SVD()
algo.fit(trainset)
print("Model Training Completed")

# Make predictions on the test set
predictions = algo.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse:.4f}")

# Function to get top N movie recommendations for a given user
def get_top_n_recommendations(predictions, user_id, n=10):
    # Get all predictions for the user
    user_predictions = [pred for pred in predictions if pred.uid == user_id]
    
    # Sort predictions by estimated rating
    user_predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get the top N movie IDs
    top_n_movie_ids = [pred.iid for pred in user_predictions[:n]]
    
    # Get movie titles
    movies = pd.read_csv('ml-latest-small/movies.csv')
    top_n_movies = movies[movies['movieId'].isin(top_n_movie_ids)]
    
    return top_n_movies[['movieId', 'title']]

# Get recommendations for a specific user
user_id = 1
top_n_movies = get_top_n_recommendations(predictions, user_id, n=10)
print("Top N Movie Recommendations for User ID 1:")
print(top_n_movies)
This full program provides a complete implementation of a movie recommendation system, including detailed print statements for each step to show the output and progress of the process.








