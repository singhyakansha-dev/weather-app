# Movie Recommendation System
A content-based movie recommendation system built using Python, Flask, Pandas, and Machine Learning techniques. The system recommends similar movies based on movie metadata and cosine similarity.

## Features
- Search movies by title
- Get top recommended similar movies
- Content-based recommendation system
- Machine Learning based similarity matching
- Interactive web interface
- Fast recommendation generation

## Technologies Used
- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- HTML
- Cosine Similarity
- Count Vectorization

## Dataset Used
The project uses the TMDB 5000 Movies Dataset containing:
- Movie titles
- Genres
- Keywords
- Cast
- Crew
- Overview

## Project Structure
```text
movie-recommendation-system/
│
├── app.py
├── index.html
├── tmdb_5000_movies.csv
├── README.md
└── requirements.txt
```

## How It Works
1. Movie metadata is processed and cleaned.
2. Important features are combined.
3. Text vectorization is performed using CountVectorizer.
4. Cosine similarity calculates similarity between movies.
5. The system recommends movies with the highest similarity scores.

## Installation
Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
```

Move into the project folder:
```bash
cd movie-recommendation-system
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the application:
```bash
python app.py
```

## Future Improvements
- Add movie posters
- Add genre filtering
- Improve recommendation accuracy
- Deploy using Render or Heroku
- Add collaborative filtering
- Add user authentication

## Author
Yakansha Singh
