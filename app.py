import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# STEP 1: Create Bigger Dataset
# -----------------------------
data = {
    'title': [
        'Batman Begins', 'The Dark Knight', 'The Dark Knight Rises',
        'Superman Returns', 'Man of Steel',
        'Spider-Man', 'Spider-Man 2', 'Spider-Man 3',
        'Iron Man', 'Iron Man 2', 'Iron Man 3',
        'Avengers', 'Avengers Age of Ultron', 'Avengers Infinity War',
        'Avengers Endgame',
        'Joker', 'Deadpool', 'Deadpool 2',
        'Thor', 'Thor Ragnarok',
        'Doctor Strange', 'Black Panther',
        'Captain America', 'Winter Soldier', 'Civil War',
        'Hulk', 'Incredible Hulk',
        'Guardians of the Galaxy', 'Guardians Vol 2',
        'Star Wars', 'Star Wars Empire Strikes Back',
        'Interstellar', 'Inception', 'Tenet',
        'The Matrix', 'Matrix Reloaded', 'Matrix Revolutions',
        'John Wick', 'John Wick 2', 'John Wick 3',
        'Fast and Furious', 'Fast Five', 'Furious 7',
        'Mission Impossible', 'Fallout', 'Ghost Protocol',
        'Harry Potter', 'Chamber of Secrets', 'Prisoner of Azkaban',
        'Goblet of Fire', 'Order of Phoenix',
        'Lord of the Rings', 'Two Towers', 'Return of the King'
    ],

    'genre': [
        'batman hero dark action', 'batman joker crime action', 'batman bane action',
        'superman hero flying', 'superman alien action',
        'spiderman hero action', 'spiderman hero action', 'spiderman venom action',
        'ironman tech hero', 'ironman tech action', 'ironman tech action',
        'avengers team hero', 'avengers robot action', 'avengers thanos war',
        'avengers endgame final battle',
        'joker crime psychology dark', 'deadpool funny action', 'deadpool action comedy',
        'thor god action', 'thor comedy action',
        'magic hero action', 'wakanda hero action',
        'captain america war hero', 'spy action hero', 'hero civil war',
        'hulk strength action', 'hulk monster action',
        'space team action', 'space team action',
        'space war sci-fi', 'space war dark',
        'space science drama', 'dream mind thriller', 'time action thriller',
        'matrix sci-fi action', 'matrix action', 'matrix war',
        'assassin action revenge', 'assassin action revenge', 'assassin action revenge',
        'car racing action', 'car heist action', 'car action',
        'spy mission action', 'spy action', 'spy action',
        'magic school fantasy', 'magic school', 'magic school dark',
        'magic tournament', 'magic rebellion',
        'fantasy ring war', 'fantasy war', 'fantasy final battle'
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# STEP 2: Combine Features
# -----------------------------
df['combined'] = df['title'] + " " + df['genre']

# -----------------------------
# STEP 3: Vectorization
# -----------------------------
cv = CountVectorizer()
vectors = cv.fit_transform(df['combined'])

# -----------------------------
# STEP 4: Similarity
# -----------------------------
similarity = cosine_similarity(vectors)

# -----------------------------
# STEP 5: Recommendation Function
# -----------------------------
def recommend(movie):
    movie = movie.lower().strip()

    matches = df[df['title'].str.lower().str.contains(movie)]

    if matches.empty:
        print("\n❌ Movie not found")
        return

    idx = matches.index[0]

    print(f"\n🎬 Showing results for: {df.iloc[idx].title}")
    print("\nTop 5 Recommended Movies:\n")

    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    for i in distances[1:6]:
        print(df.iloc[i[0]].title)

# -----------------------------
# STEP 6: Show sample movies
# -----------------------------
print("\nAvailable Movies:")
print(df['title'].head(20))

# -----------------------------
# STEP 7: Input Loop
# -----------------------------
while True:
    movie = input("\nEnter movie name (or 'exit'): ")

    if movie.lower() == 'exit':
        print("👋 Exiting...")
        break

    recommend(movie)