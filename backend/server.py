from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from flask_sqlalchemy import SQLAlchemy
import requests
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
import io
import logging
import threading
import random
from googleapiclient.errors import HttpError
import logging
from sqlalchemy.dialects.postgresql import JSONB
from werkzeug.security import generate_password_hash, check_password_hash # <--- Add these imports
import datetime
from sqlalchemy import func # For db.func.max
from tqdm import tqdm
import torch
from model_utils import load_model, CandidateGenerationModel # Assuming this import is correct
import torch.nn.functional as F
import difflib
import os
import google.generativeai as genai
from dotenv import load_dotenv # If using .env file (pip install python-dotenv)
import praw

app = Flask(__name__)
CORS(app)

# Configure the PostgreSQL database here
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost:5432/book'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu" # Or "cuda" if available and configured
print("Using device:", device)
# In server.py (or app.py)

# Load user/book index mappings
try:
    with open("models/user_book_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
        user2idx = mappings.get('user2idx')          # keys: original user IDs
        book2idx = mappings.get('book2idx')          # keys: original book IDs (e.g., Goodreads IDs)
        # --- << NEW: Create reverse mapping for index to original book ID >> ---
        # We might need this if you want to return original IDs instead of internal indices
        idx2book = {idx: book_id for book_id, idx in book2idx.items()}
        # ----------------------------------------------------------------------

        if user2idx is None or book2idx is None:
            raise ValueError("Mappings file missing 'user2idx' or 'book2idx' keys.")
        print(f"Loaded mappings for {len(user2idx)} users and {len(book2idx)} books.")

except FileNotFoundError:
    print("ERROR: Mapping file 'models/user_book_mappings.pkl' not found.")
    exit() # Or handle more gracefully
except Exception as e:
    print(f"ERROR: Failed to load mappings: {e}")
    exit()


# --- << NEW: Load books_data.csv and create a mapping for book ID to title >> ---
try:
    data_books = pd.read_csv("books_data.csv")
    book_id_to_title = dict(zip(data_books['Id'].astype(str), data_books['Title']))
    print(f"Loaded book data for {len(book_id_to_title)} books.")
except FileNotFoundError:
    print("ERROR: Book data file 'books_data.csv' not found.")
    book_id_to_title = {} # Initialize as empty if not found
except Exception as e:
    print(f"ERROR: Failed to load book data: {e}")
    book_id_to_title = {} # Initialize as empty if loading fails
# -----------------------------------------------------------------------------

# Load trained model
try:
    num_users = len(user2idx)
    num_books = len(book2idx)
    if num_users == 0 or num_books == 0:
        raise ValueError("No users or books found in mappings.")
    # Assuming load_model takes model path, num_users, num_books
    model = load_model("models/candidate_model.pt", num_users, num_books).to(device)
    model.load_state_dict(torch.load('models/candidate_model.pt'), strict=False)
    model.eval() # Set model to evaluation mode
    print("Successfully loaded recommendation model.")
except FileNotFoundError:
    print("ERROR: Model file 'models/candidate_model.pt' not found.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    exit()

def generate_and_save_playlist_recommendations(user_id):
    """
    Generates recommendations based on similarity to the user's playlist items
    using loaded model embeddings (via title mapping), fetches details via
    Google Books API, and saves results to the 'recommend' table.
    Must be run within an app context (e.g., wrapped by with app.app_context()).
    """
    with app.app_context():
        app.logger.info(
            f"Generating item-based recommendations for user_id: {user_id} (using title mapping)"
        )

        # 1) Load user
        user = db.session.get(User, user_id)
        if not user:
            app.logger.warning(f"User not found: {user_id}")  # Changed warn to warning
            return {'status': 'fail', 'message': 'User not found.'}

        # 2) Fetch playlist items & map titles to indices
        try:
            playlist_items = (
                db.session.query(Book)
                .join(playlist_books, Book.id == playlist_books.c.book_id)
                .filter(playlist_books.c.user_id == user_id)
                .order_by(playlist_books.c.position)
                .limit(25)
                .all()
            )
            playlist_google_ids = {
                book.google_book_id
                for book in playlist_items
                if book.google_book_id
            }

            playlist_book_idxs = []
            valid_playlist_titles = []
            
            titles = data_books['Title'].dropna().astype(str).tolist()
            for book in playlist_items:
                title = book.title
                closest_match = difflib.get_close_matches(title, titles, n=1)
                if closest_match:
                    matched_title = closest_match[0]
                    title = data_books.loc[data_books['Title'] == matched_title, 'Id'].values[0]
                if title and title in book2idx:
                    playlist_book_idxs.append(book2idx[title])
                    valid_playlist_titles.append(title)
                elif title:
                    app.logger.warning(  # Changed warn to warning
                        f"Playlist book title '{title}' not in book2idx mapping."
                    )
        except Exception as e:
            app.logger.error(
                f"Error fetching playlist/mapping for user {user_id}: {e}",
                exc_info=True
            )
            return {'status': 'fail', 'message': 'Error processing user playlist.'}

        if not playlist_book_idxs:
            app.logger.info(
                f"User {user_id} has no playlist items with titles in model mapping. Clearing recommendations."
            )
            try:
                #stmt_del = recommend.delete().where(
                #   recommend.c.user_id == user_id
                #)
                #db.session.execute(stmt_del)
                #db.session.commit()
                return {
                    'status': 'success',
                    'message': 'No relevant playlist items found; cleared recommendations.',
                    'count': 0
                }
            except Exception as e:
                db.session.rollback()
                app.logger.error(f"Error clearing recs: {e}")
                return {'status': 'fail', 'message': 'Error clearing recommendations.'}

        app.logger.info(
            f"Found {len(playlist_book_idxs)} valid playlist seeds (titles) for user {user_id}: {valid_playlist_titles}"
        )
        

        # 3) Generate recommendations using item embeddings
        potential_recommendations = {}
        try:
            if model is None:
                raise ValueError("Recommendation model is not loaded.")
            
            # No need to load state dict again
            # model.load_state_dict(torch.load("models/candidate_model.pt"))
            
            # Generate embeddings for all books in your index
            book_titles = [idx2book.get(idx) for idx in range(len(idx2book))]
            
            # Get book titles from IDs (you may need to map IDs to titles first)
            book_titles = [book_id_to_title.get(book_id, "") for book_id in book_titles]
            
            # Filter out empty titles
            valid_idx_to_title = {idx: title for idx, title in enumerate(book_titles) if title}
            
            # Generate embeddings
            all_book_embeddings = {}
            batch_size = 32  # Adjust based on your memory constraints
            
            for batch_start in range(0, len(valid_idx_to_title), batch_size):
                batch_indices = list(valid_idx_to_title.keys())[batch_start:batch_start+batch_size]
                batch_titles = [valid_idx_to_title[idx] for idx in batch_indices]
                
                # Generate embeddings using SentenceTransformer
                batch_embeddings = model.encode(batch_titles, convert_to_tensor=True, device=device)
                
                # Store embeddings with their indices
                for idx, embedding in zip(batch_indices, batch_embeddings):
                    all_book_embeddings[idx] = embedding
            
            # Convert to tensor for similarity calculation
            indices = list(all_book_embeddings.keys())
            embeddings = torch.stack([all_book_embeddings[idx] for idx in indices])
            
            app.logger.info("Calculating similarities based on playlist item embeddings...")
            
            for seed_idx in playlist_book_idxs:
                if seed_idx not in all_book_embeddings:
                    continue
                    
                seed_vec = all_book_embeddings[seed_idx].unsqueeze(0)
                # Already normalized by the model's Normalize() layer
                
                # Calculate similarities
                sims = torch.matmul(seed_vec, embeddings.t()).squeeze(0)
                
                # take top k per seed
                k = min(25, sims.size(0))
                top_scores, top_idxs = torch.topk(sims, k)
                
                for score, tensor_idx in zip(top_scores.tolist(), top_idxs.tolist()):
                    idx = indices[tensor_idx]
                    if idx == seed_idx:
                        continue
                    gid = idx2book.get(idx)
                    if not gid or gid in playlist_google_ids:
                        continue
                    prev = potential_recommendations.get(gid, 0)
                    if score > prev:
                        potential_recommendations[gid] = score
                        
            app.logger.info(
                f"Generated {len(potential_recommendations)} unique candidates."
            )
        except Exception as e:
            app.logger.error(
                f"Error during similarity calculation: {e}",
                exc_info=True
            )
            return {'status': 'fail', 'message': 'Error during model processing.'}

        # 4) Sort and limit candidates
        sorted_cands = sorted(
            potential_recommendations.items(),
            key=lambda item: item[1],
            reverse=True
        )
        top_ids = [gid for gid, _ in sorted_cands[:10]]

        if not top_ids:
            app.logger.info(
                f"No new recommendations found for user {user_id}. Clearing recs."
            )
            #stmt_del = recommend.delete().where(
            #    recommend.c.user_id == user_id
            #)
            #db.session.execute(stmt_del)
            #db.session.commit()
            return {'status': 'success', 'message': 'No recommendations generated.', 'count': 0}

        # 5) Enrich candidates with DB/Google API
        final_recs = []
        for gid in top_ids:
            # Get the title from your data_books DataFrame
            try:
                title = data_books.loc[data_books['Id'] == gid, 'Title'].values[0]
            except (IndexError, KeyError):
                app.logger.warning(f"No title found for ID {gid}. Skipping.")  # Changed warn to warning
                continue
                
            # First, try to find the book in the database by title or google_book_id
            book_obj = db.session.query(Book).filter(
                (Book.google_book_id == gid) | (Book.title.ilike(f"%{title}%"))
            ).first()
            
            if book_obj:
                final_recs.append(book_obj.to_dict())
            else:
                try:
                    # Search for the book on Google Books API using the title
                    query_params = {
                        'q': f'intitle:{title}',
                        'maxResults': 1
                    }
                    search_response = service.volumes().list(**query_params).execute()
                    items = search_response.get('items', [])
                    
                    if items:
                        item = items[0]
                        item_gid = item.get('id')
                        info = item.get('volumeInfo', {})
                        sale = item.get('saleInfo', {})
                        img = info.get('imageLinks', {})
                        
                        final_recs.append({
                            'id': None,
                            'google_book_id': item_gid or gid,  # Use the found ID or original ID
                            'title': info.get('title', title),  # Use original title if not found
                            'authors': ", ".join(info.get('authors', [])),
                            'genre': (info.get('categories') or [''])[0],
                            'synopsis': info.get('description', ''),
                            'rating': info.get('averageRating'),
                            'image_link': img.get('thumbnail'),
                            'listPrice': sale.get('listPrice', {}).get('amount'),
                            'buyLink': sale.get('buyLink')
                        })
                    else:
                        app.logger.warning(f"No Google Books results found for title: {title}")  # Changed warn to warning
                        
                except Exception as e:
                    app.logger.error(f"Error searching title '{title}': {e}")
            
            if len(final_recs) >= 20:
                break
        # 6) Save final recommendations
        count = 0
        try:
            # First, get all existing recommendations for this user to avoid duplicates
            # Using the SQLAlchemy execute directly with SQL expression
            existing_recs = db.session.execute(
                db.text("SELECT book_id FROM recommend WHERE user_id = :user_id"),
                {"user_id": user_id}
            ).fetchall()
            existing_book_ids = {rec[0] for rec in existing_recs}
            
            # Delete existing recommendations for this user
            #stmt_del = recommend.delete().where(
            #    recommend.c.user_id == user_id
            #)
            #db.session.execute(stmt_del)
            
            # Track books we've already added to avoid duplicates
            added_book_ids = set()
            
            for pos, rec in enumerate(final_recs, start=1):
                gid = rec.get('google_book_id')
                
                # Skip items without a Google Book ID
                if not gid:
                    continue
                    
                # Check if we already have this book in the database
                book_obj = db.session.query(Book).filter_by(
                    google_book_id=gid
                ).first()
                
                if not book_obj:
                    # Create new Book if needed
                    data = {
                        'google_book_id': rec.get('google_book_id'),
                        'title': rec.get('title'),
                        'authors': rec.get('authors'),
                        'genre': rec.get('genre'),
                        'synopsis': rec.get('synopsis'),
                        'rating': rec.get('rating'),
                        'image_link': rec.get('image_link')
                    }
                    book_obj = Book(**data)
                    db.session.add(book_obj)
                    db.session.flush()
                
                # Only add if we have a valid book and haven't already added it
                if book_obj and book_obj.id and book_obj.id not in added_book_ids:
                    db.session.execute(
                        recommend.insert().values(
                            user_id=user_id,
                            book_id=book_obj.id,
                            position=pos
                        )
                    )
                    added_book_ids.add(book_obj.id)
                    count += 1
                    
            db.session.commit()
            app.logger.info(f"Saved {count} recommendations for user {user_id}.")
            return {'status': 'success', 'message': f'Generated & saved {count} recommendations.', 'count': count}
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error saving recs: {e}", exc_info=True)
            return {'status': 'fail', 'message': 'Error saving recommendations.'}
# End of function

# Association tables for liked books and playlist books
liked_books = db.Table('liked_books',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('book_id', db.Integer, db.ForeignKey('books.id'), primary_key=True)
)

playlist_books = db.Table('playlist_books',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('book_id', db.Integer, db.ForeignKey('books.id'), primary_key=True),
    db.Column('position', db.Integer, nullable=False, default=0),  # Added position column
    db.Column('tag', db.String(50), nullable=False, default='save_later')
)

recommend = db.Table('recommend',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('book_id', db.Integer, db.ForeignKey('books.id'), primary_key=True),
    db.Column('position', db.Integer, nullable=False, default=0),  # Added position column
)

# Define the User model
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)

    # --- << CHANGE: Store hash instead of plaintext >> ---
    # Rename column and increase length for the hash
    password_hash = db.Column(db.String(128), nullable=False)
    # -----------------------------------------------------

    created_at = db.Column(db.DateTime, server_default=db.func.now())
    # Add the preferences flag if you haven't already from previous steps
    has_selected_preferences = db.Column(db.Boolean, default=False, nullable=False)

    # Relationships (Keep as they were)
    liked_books = db.relationship('Book', secondary=liked_books, backref=db.backref('liked_by', lazy='dynamic'))
    playlist_books = db.relationship('Book', secondary=playlist_books, backref=db.backref('playlist_for', lazy='dynamic'))

    # --- << ADD: Password Hashing Methods >> ---
    def set_password(self, password):
        """Create hashed password."""
        # Hashes the password with a random salt (default method)
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256') # Explicitly choose a strong method

    def check_password(self, password):
        """Check hashed password."""
        # Checks the provided password against the stored hash
        return check_password_hash(self.password_hash, password)
    # ---------------------------------------

    def to_dict(self):
        """Returns user data as a dictionary suitable for JSON serialization."""
        # Make sure this method exists from previous steps
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'has_selected_preferences': self.has_selected_preferences
            # --- IMPORTANT: NEVER return password_hash here ---
        }

    def __repr__(self):
        return f'<User {self.username}>'

# Define the Book model
class Book(db.Model):
    __tablename__ = 'books'
    id = db.Column(db.Integer, primary_key=True)
    google_book_id = db.Column(db.String(255), unique=True, nullable=True)
    title = db.Column(db.String(255))
    authors = db.Column(db.String(255))
    genre = db.Column(db.String(255))  # Added genre field
    synopsis = db.Column(db.Text)
    rating = db.Column(db.Float)
    image_link = db.Column(db.String(255))

    def __repr__(self):
        return f'<Book {self.title}>'

    def to_dict(self):
        """Convert book object to dictionary for JSON response"""
        return {
            'id': self.id,
            'google_book_id': self.google_book_id,
            'title': self.title,
            'authors': self.authors,
            'genre': self.genre,  # Added genre to the dictionary
            'synopsis': self.synopsis,
            'rating': self.rating,
            'image_link': self.image_link
        }
    

class CachedBookDetail(db.Model):
    __tablename__ = 'cached_book_details' # Choose a table name

    # Use google_book_id as the primary key for simplicity in this cache table
    google_book_id = db.Column(db.String(255), primary_key=True)
    title = db.Column(db.String(255), nullable=True)
    authors = db.Column(db.String(255), nullable=True)
    genre = db.Column(db.String(255), nullable=True)
    synopsis = db.Column(db.Text, nullable=True)
    rating = db.Column(db.Float, nullable=True)
    image_link = db.Column(db.String(255), nullable=True)
    # Use JSON type for listPrice (dictionary) and buyLink (string)
    listPrice = db.Column(JSONB, nullable=True) # Or db.JSON
    buyLink = db.Column(db.String(1024), nullable=True) # Store the URL
    # Optional: Add a timestamp for when it was cached
    cached_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())

    def to_dict(self):
        """Convert cached detail object to dictionary for JSON response"""
        return {
            # Use google_book_id for both id fields for consistency with frontend expectation
            'id': self.google_book_id,
            'google_book_id': self.google_book_id,
            'title': self.title,
            'authors': self.authors,
            'genre': self.genre,
            'synopsis': self.synopsis,
            'rating': self.rating,
            'image_link': self.image_link,
            'listPrice': self.listPrice, # Will be the dict/null retrieved from JSON column
            'buyLink': self.buyLink
        }

    def __repr__(self):
        return f'<CachedBookDetail {self.google_book_id}: {self.title}>'


def load_api_keys(filepath):
    keys = {}
    with open(filepath, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                keys[key.strip()] = value.strip()
    return keys

# Load from file
api_keys = load_api_keys('password.txt')

gemini_key = api_keys.get('gemini_api_key')
service_1 = api_keys.get('service_1')
service_2 = api_keys.get('service_2')
reddit_client_id = api_keys.get('reddit_client_id')
reddit_client_secret = api_keys.get('reddit_cliend_secret')

# Initialize the Books API service
#service = build('books', 'v1', developerKey=service_1, cache_discovery=False)
service = build('books', 'v1', developerKey=service_2,cache_discovery=False)
    
load_dotenv() # Load environment variables from .env file
GEMINI_API_KEY = gemini_key


if not GEMINI_API_KEY:
    app.logger.warn("GOOGLE_API_KEY environment variable not set.")
    # Handle missing key appropriately - maybe disable Gemini features
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        app.logger.info("Gemini API Key configured.")
    except Exception as e:
         app.logger.error(f"Error configuring Gemini API: {e}")
     
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", reddit_client_id) # Replace with your client ID
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", reddit_client_secret) # Replace with your client secret
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "script:my-book-review-app:v1.0 (by /u/your_username)") # Replace with your user agent

# --- Configuration ---
# Subreddits to search within (can be adjusted)
DEFAULT_SUBREDDITS = [
    'books', 'literature', 'suggestmeabook', 'booksuggestions',
    'Fantasy', 'SciFi', 'freeEbooks', '52book', 'whatsthatbook',
    'bookshelf', 'Outlander', 'YAlit', 'bookexchange',
]
LIMIT_PER_SUBREDDIT = 5 # Limit results per subreddit for API performance

# --- Helper Function (Adapted from your script) ---
def get_reddit_reviews_internal(book_title, subreddits=None, limit=LIMIT_PER_SUBREDDIT):
    """
    Internal function to fetch Reddit posts.
    Uses credentials loaded from environment variables.
    """
    if subreddits is None:
        subreddits = DEFAULT_SUBREDDITS

    reviews = []

    # Check if credentials are set
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or not REDDIT_USER_AGENT:
        print("ERROR: Reddit API credentials not found in environment variables.")
        # Return an indicator of configuration error, or raise an exception
        # For the API route, we'll return an error response later.
        return None # Indicate configuration error

    try:
        print(f"Initializing PRAW with User Agent: {REDDIT_USER_AGENT}")
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            read_only=True # Explicitly set read-only mode
        )
        print(f"PRAW Initialized. Read-Only: {reddit.read_only}")

        # Construct the search query
        # Using quotes helps find exact matches. Added "review" to narrow down.
        search_query = f'"{book_title}" book review'
        print(f"Searching Reddit for: '{search_query}' in subreddits: {', '.join(subreddits)}")

        for sub_name in subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)
                print(f"-> Searching in r/{sub_name}...")

                search_results = subreddit.search(search_query, sort='relevance', limit=limit)

                found_count = 0
                for submission in search_results:
                    # Basic filtering: Check if title is relevant
                    if book_title.lower() in submission.title.lower() or \
                       (submission.selftext and book_title.lower() in submission.selftext.lower()):

                        reviews.append({
                            'title': submission.title,
                            'score': submission.score,
                            'url': submission.url,
                            'subreddit': sub_name,
                            'body_snippet': submission.selftext[:200] + '...' if submission.selftext else '[No body text]', # Slightly longer snippet
                            'created_utc': submission.created_utc # Add creation time
                        })
                        found_count += 1
                if found_count > 0:
                     print(f"   Found {found_count} potential post(s) in r/{sub_name}.")

            except praw.exceptions.PRAWException as e:
                print(f"   Error searching subreddit r/{sub_name}: {e}")
            except Exception as e:
                 print(f"   An unexpected error occurred while searching r/{sub_name}: {e}")

    except praw.exceptions.PRAWException as e:
        print(f"ERROR: Failed to connect to Reddit or PRAW error: {e}")
        raise # Re-raise PRAW exceptions to be caught by the route handler
    except Exception as e:
        print(f"An unexpected error occurred during Reddit search: {e}")
        raise # Re-raise other exceptions

    # Sort reviews by score (descending)
    reviews.sort(key=lambda x: x['score'], reverse=True)

    return reviews

# --- New API Route ---
@app.route('/api/reddit-reviews', methods=['GET'])
def fetch_reddit_reviews_route():
    """
    API endpoint to fetch Reddit posts related to a book title.
    Expects 'title' as a query parameter.
    e.g., /api/reddit-reviews?title=Project%20Hail%20Mary
    """
    book_title = request.args.get('title')

    if not book_title:
        return jsonify({"error": "Missing 'title' query parameter"}), 400

    # Check credentials before proceeding
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or not REDDIT_USER_AGENT:
         return jsonify({"error": "Reddit API credentials not configured on server."}), 500

    try:
        print(f"Received request for Reddit reviews for title: {book_title}")
        reviews_data = get_reddit_reviews_internal(book_title)

        if reviews_data is None:
             # This case now specifically means credentials weren't found by the helper
             return jsonify({"error": "Reddit API credentials configuration error on server."}), 500

        print(f"Found {len(reviews_data)} relevant Reddit posts for '{book_title}'.")
        return jsonify(reviews_data), 200

    except praw.exceptions.ResponseException as e:
         # Handle specific PRAW response errors (e.g., 401 Unauthorized, 403 Forbidden)
         print(f"PRAW Response Error fetching Reddit reviews: {e}")
         return jsonify({"error": f"Reddit API Error: {e.response.status_code} - Check credentials or permissions."}), 500
    except praw.exceptions.PRAWException as e:
        print(f"PRAW Error fetching Reddit reviews: {e}")
        return jsonify({"error": "Failed to fetch data from Reddit due to PRAW error."}), 500
    except Exception as e:
        print(f"Unexpected Server Error fetching Reddit reviews: {e}")
        # Log the full error traceback here for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An unexpected server error occurred."}), 500

         
@app.route('/api/user/<int:user_id>/ask_gemini', methods=['POST'])
# @login_required # Recommended for security
def ask_gemini_about_books(user_id):
    # --- Basic User Check ---
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    # --- Get User Query ---
    data = request.get_json()
    user_query = data.get('query')
    if not user_query:
        return jsonify({'error': 'Missing query in request body'}), 400

    # --- Fetch User's Books (Example: Playlist + Liked, limit total) ---
    books_context = []
    try:
        # Get Playlist Books (limit to ~15)
        playlist_items = (
            db.session.query(Book.title, Book.authors, Book.genre)
            .join(playlist_books, Book.id == playlist_books.c.book_id)
            .filter(playlist_books.c.user_id == user_id)
            .order_by(playlist_books.c.position)
            .limit(15).all() )
        if playlist_items:
             books_context.append("Playlist Books:\n" + "\n".join([f"- '{b.title}' by {b.authors or 'Unknown'} (Genre: {b.genre or 'N/A'})" for b in playlist_items]))

        # Get Liked Books (limit to ~15, avoid duplicates)
        playlist_titles = {b.title for b in playlist_items if b.title} # Titles already included
        liked_items = (
             db.session.query(Book.title, Book.authors, Book.genre)
             .join(liked_books, Book.id == liked_books.c.book_id)
             .filter(liked_books.c.user_id == user_id)
             .filter(Book.title.notin_(playlist_titles)) # Avoid adding duplicates already listed from playlist
             .limit(15).all() )
        if liked_items:
             books_context.append("Liked Books:\n" + "\n".join([f"- '{b.title}' by {b.authors or 'Unknown'} (Genre: {b.genre or 'N/A'})" for b in liked_items]))

    except Exception as e:
        app.logger.error(f"Error fetching books for Gemini context (User {user_id}): {e}")
        return jsonify({'error': 'Could not retrieve user book data'}), 500

    if not books_context:
        context_string = "The user currently has no books listed in their playlist or liked books."
    else:
        context_string = "\n\n".join(books_context)
    # --- Construct the Prompt ---
    prompt = f"""
    You are a helpful and intelligent book recommendation assistant.

    Your goal is to recommend books based on the user's question. If the user's book list contains relevant genres or titles, you may use that as inspiration. However, if the list does not help, feel free to use your general book knowledge to answer.

    User's Book List:
    ---
    {context_string}
    ---

    User's Question: "{user_query}"

    Answer:
    """

    app.logger.info(f"Sending prompt to Gemini for user {user_id}")
    # app.logger.debug(f"Prompt: {prompt}") # Log prompt for debugging if needed

    # --- Call Gemini API ---
    try:
        if not GEMINI_API_KEY: # Check again if key was loaded
             return jsonify({'error': 'Gemini API key not configured on server.'}), 500

        # Choose a model (gemini-1.5-flash is often faster/cheaper for simple Q&A)
        model = genai.GenerativeModel('gemini-2.0-flash')
        # Or use gemini-pro for potentially more complex reasoning:
        # model = genai.GenerativeModel('gemini-pro')

        # Generate content
        # Add safety settings if desired
        # safety_settings=[...]
        response = model.generate_content(prompt) #, safety_settings=safety_settings)

        # Log the full response for debugging if needed
        # app.logger.debug(f"Gemini Raw Response: {response}")

        # --- Process Response ---
        # Check for blocked content due to safety filters
        if not response.candidates:
             app.logger.warn(f"Gemini response for user {user_id} blocked or empty. Prompt: {prompt[:200]}...") # Log truncated prompt
             # Check response.prompt_feedback for block reason
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
             return jsonify({'answer': f"Sorry, I couldn't generate a response for that query. (Reason: {block_reason})"})

        # Extract text - handle potential errors if structure is unexpected
        generated_text = response.text

        app.logger.info(f"Received Gemini response for user {user_id}")
        return jsonify({'answer': generated_text})

    except Exception as e:
        app.logger.error(f"Error calling Gemini API for user {user_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to get response from AI assistant: {e}'}), 500

        
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("book_index.faiss")
metadata = pd.read_pickle("book_metadata.pkl")
def search_books(query_title, query_authors, query_genre, query_synopsis, top_k=10):
    """
    Searches for similar books based on the combined query provided by title, authors, genre, and synopsis.
    Returns a list of dictionaries for the top similar books.
    """
    
    # Create the combined text for the query
    query_text = f"{query_title} by {query_authors}. Genre: {query_genre}. Synopsis: {query_synopsis}"

    # Generate embedding for the query and normalize it
    query_embedding = model.encode([query_text], show_progress_bar=True)
    query_embedding = normalize(query_embedding, axis=1)

    # Search the FAISS index; top_k results
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)

    # Collect the top similar books
    results = []
    for i, idx in enumerate(indices[0]):
        candidate = metadata.iloc[idx]
        results.append({
            'title': candidate['title'],
            'authors': candidate['authors'],
            'genre': candidate['genre'],
            'synopsis': candidate['synopsis'],
            'num_ratings': int(candidate['num_ratings']),
            'num_reviews': int(candidate['num_reviews']),
            'similarity': float(distances[0][i])
        })

    return results

@app.route('/data', methods=['GET'])
def data(num_results=10):       
    response = service.volumes().list(
        q='subject:fiction',
        orderBy='relevance',
        maxResults=num_results,
    ).execute()
    books_list = []
    if 'items' in response:
        for item in response['items']:
            volume_info = item['volumeInfo']
            image_links = volume_info.get('imageLinks', {})
            # Attempt to get the best available image
            image_url = (
                image_links.get('extraLarge') or
                image_links.get('large') or
                image_links.get('medium') or
                image_links.get('thumbnail', 'N/A')
            )
            
            # Extract genre information from categories field
            categories = volume_info.get('categories', [])
            genre = ', '.join(categories) if categories else 'N/A'
            
            books_list.append({
                'id': item.get('id'),  # Changed to 'id' to match React component
                'google_book_id': item.get('id'),
                'title': volume_info.get('title', 'N/A'),
                'authors': ', '.join(volume_info.get('authors', ['N/A'])),
                'genre': genre,  # Added genre
                'synopsis': volume_info.get('description', 'N/A'),
                'rating': volume_info.get('averageRating', 'N/A'),
                'image_link': image_url
            })
    return jsonify(books_list)

@app.route('/user/<int:user_id>/info', methods=['GET'])
def user_info(user_id):
    """Fetches stored recommendations for a user, excluding books without image links."""
    app.logger.info(f"Fetching recommendations for user_id: {user_id}")
    try:
        # Get the user's recommendations sorted by position
        recommendation_items = db.session.query(Book, recommend.c.position)\
            .join(recommend, Book.id == recommend.c.book_id)\
            .filter(recommend.c.user_id == user_id)\
            .order_by(recommend.c.position).all()

        books_list = []
        skipped_count = 0
        for book, position in recommendation_items:
            # --- << ADD CHECK FOR image_link >> ---
            # Check if image_link exists (is not None, not empty string)
            # AND explicitly check it's not the placeholder string 'N/A' if that might be stored
            if book.image_link and book.image_link != 'N/A':
                # If image link is valid, create and append the dictionary
                books_list.append({
                    'id': book.google_book_id,
                    'google_book_id': book.google_book_id,
                    'title': book.title or 'N/A',
                    'authors': book.authors or 'N/A',
                    'genre': book.genre or 'N/A',
                    'synopsis': book.synopsis or 'N/A',
                    'rating': book.rating if book.rating is not None else 'N/A',
                    'image_link': book.image_link, # Use the valid link
                    'position': position,
                    'tag': 'recommend'
                })
            else:
                # Optional: Log skipped books for debugging
                skipped_count += 1
                app.logger.debug(f"Skipping book recommendation (User: {user_id}, Title: {book.title}) due to missing/invalid image_link: {book.image_link}")
            # --- << END CHECK >> ---

        if skipped_count > 0:
             app.logger.info(f"Skipped {skipped_count} recommendations for user {user_id} due to missing image links.")

        # Random sampling logic (remains the same)
        if len(books_list) > 30:
            app.logger.info(f"Returning 30 random samples from {len(books_list)} valid recommendations for user {user_id}.")
            final_books_list = random.sample(books_list, 30)
        else:
            app.logger.info(f"Returning all {len(books_list)} valid recommendations for user {user_id}.")
            final_books_list = books_list

        # Return the filtered result as JSON
        return jsonify(final_books_list)

    except Exception as e:
        app.logger.error(f"Error fetching recommendations for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch recommendations"}), 500



def add_recommendations(user_id):
    with app.app_context():
        # Check if the user exists.
        user = User.query.get(user_id)
        if not user:
            return jsonify({'status': 'fail', 'message': 'User not found'}), 404

        # Get the user's playlist items.
        playlist_items = (
            db.session.query(Book, playlist_books.c.position, playlist_books.c.tag)
            .join(playlist_books, Book.id == playlist_books.c.book_id)
            .filter(playlist_books.c.user_id == user_id)
            .order_by(playlist_books.c.position)
            .all()
        )

        # Build a list of book dictionaries from the playlist.
        books_list = []
        for book, position, tag in playlist_items:
            book_dict = book.to_dict()
            book_dict['position'] = position
            book_dict['tag'] = tag
            books_list.append(book_dict)

        # Generate recommendations using your search_books function.
        recommended_books = []
        for book in books_list:
            # For each book in the playlist, retrieve similar books.
            similar_books = search_books(
                query_title=book['title'],
                query_authors=book['authors'],
                query_genre=book['genre'],
                query_synopsis=book['synopsis'],
                top_k=5  # Adjust the number as needed.
            )
            for similar in similar_books:
                sim_query = similar['title']
                sim_response = service.volumes().list(
                    q=sim_query,
                    orderBy='relevance',
                    maxResults=1,
                ).execute()

                if 'items' in sim_response:
                    for sim_item in sim_response['items']:
                        sim_volume_info = sim_item['volumeInfo']
                        image_links = sim_volume_info.get('imageLinks', {})
                        image_url = (
                            image_links.get('extraLarge') or
                            image_links.get('large') or
                            image_links.get('medium') or
                            image_links.get('thumbnail', 'N/A')
                        )
                        # Use the external API's id for the google_book_id.
                        ext_book_id = sim_item.get('id')
                        # Only add if this recommendation is not already present.
                        if not any(r.get('google_book_id') == ext_book_id for r in recommended_books):
                            sim_categories = sim_volume_info.get('categories', ['N/A'])
                            sim_genre = sim_categories[0] if sim_categories else 'N/A'
                            recommended_books.append({
                                'google_book_id': ext_book_id,
                                'title': sim_volume_info.get('title', 'N/A'),
                                'authors': ', '.join(sim_volume_info.get('authors', ['N/A'])),
                                'genre': sim_genre,
                                'synopsis': sim_volume_info.get('description', 'N/A'),
                                'rating': sim_volume_info.get('averageRating', 'N/A'),
                                'image_link': image_url,
                                'similarity': similar.get('similarity', 0)
                            })  
        #deleting books in playlist                    
        db.session.execute(
        recommend.delete().where(recommend.c.user_id == user_id)
        )
        db.session.commit()
        # For each recommended book, insert a row in the "recommend" table.
        recommendations_added = 0
        for rec in recommended_books:
            # Check if the book already exists in the Book table; if not, create it.
            book_obj = Book.query.filter_by(google_book_id=rec['google_book_id']).first()
            if not book_obj:
                try:
                    rating_value = float(rec['rating']) if rec['rating'] not in [None, '', 'N/A'] else None
                except (ValueError, TypeError):
                    rating_value = None
                book_obj = Book(
                    google_book_id=rec['google_book_id'],
                    title=rec['title'],
                    authors=rec['authors'],
                    genre=rec['genre'],
                    synopsis=rec['synopsis'],
                    rating=rating_value,
                    image_link=rec['image_link']
                )
                db.session.add(book_obj)
                db.session.commit()  # Commit to generate the book_obj.id

            # Check if this recommendation already exists for the user.
            existing = db.session.execute(
                recommend.select().where(
                    (recommend.c.user_id == user_id) &
                    (recommend.c.book_id == book_obj.id)
                )
            ).fetchone()
            if existing:
                continue

            # Update the max position query - this is the correct syntax for SQLAlchemy
            max_position_result = db.session.query(db.func.max(recommend.c.position))\
                .filter(recommend.c.user_id == user_id).scalar()
            next_position = 1 if max_position_result is None else max_position_result + 1

            # Insert into the recommendations (recommend) table.
            stmt = recommend.insert().values(
                user_id=user_id,
                book_id=book_obj.id,
                position=next_position
            )
            db.session.execute(stmt)
            recommendations_added += 1

        db.session.commit()
        return jsonify({
            'status': 'success',
            'message': 'Recommendations added',
            'count': recommendations_added
        })
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
#this function is useless now wrote this for testing 
""" 
def run_background_recommendations_for_user(user_id):
    with app.app_context():
        try:
            logger.info(f"Starting background recommendation generation for user_id: {user_id}")

            user = db.session.query(User).filter(User.id == user_id).first()
            if not user:
                logger.error(f"User with id {user_id} not found.")
                return

            # --- Call First Recommendation Function ---
            try:
                logger.info(f" -> Calling add_recommendations for user {user.id}...")
                result1 = add_recommendations(user.id)
                if result1 and result1.get('status') == 'success':
                    logger.info(f"    add_recommendations result: {result1.get('message')} (Count: {result1.get('count')})")
                else:
                    logger.error(f"    add_recommendations failed for user {user.id}: {result1.get('message') if result1 else 'Unknown error'}")
            except Exception as e1:
                logger.error(f"    Critical error calling add_recommendations for user {user.id}: {e1}", exc_info=True)

            # --- Call Second Recommendation Function ---
            try:
                logger.info(f" -> Calling generate_and_save_playlist_recommendations for user {user.id}...")
                #result2 = generate_and_save_playlist_recommendations(user.id, db)
                #if result2 and result2.get('status') == 'success':
                #   logger.info(f"    generate_... result: {result2.get('message')} (Count: {result2.get('count')})")
                #else:
                #    logger.error(f"    generate_... failed for user {user.id}: {result2.get('message') if result2 else 'Unknown error'}")
            except Exception as e2:
                logger.error(f"    Critical error calling generate_... for user {user.id}: {e2}", exc_info=True)

            logger.info(f"Finished processing recommendations for user {user.id}.")
        except Exception as e:
            logger.error(f"Error in background recommendation process for user_id {user_id}: {e}", exc_info=True)
"""  


@app.route('/search', methods=['GET'])
def search():
    # Require the client to provide a query, no default subject filter
    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    # Get num_results from parameters, default to 10
    num_results = request.args.get('num_results', 10, type=int)
    
    response = service.volumes().list(
        q=query,
        orderBy='relevance',
        maxResults=num_results,
    ).execute()
    
    books_list = []
    if 'items' in response:
        for item in response['items']:
            volume_info = item['volumeInfo']
            image_links = volume_info.get('imageLinks', {})
            # Attempt to get the best available image
            image_url = (
                image_links.get('extraLarge') or
                image_links.get('large') or
                image_links.get('medium') or
                image_links.get('thumbnail', 'N/A')
            )
            
            # Extract genre information from categories field
            categories = volume_info.get('categories', [])
            genre = ', '.join(categories) if categories else 'N/A'
            
            books_list.append({
                'id': item.get('id'),  # Unique identifier
                'google_book_id': item.get('id'),
                'title': volume_info.get('title', 'N/A'),
                'authors': ', '.join(volume_info.get('authors', ['N/A'])),
                'genre': genre,  # Added genre
                'synopsis': volume_info.get('description', 'N/A'),
                'rating': volume_info.get('averageRating', 'N/A'),
                'image_link': image_url
            })
    return jsonify(books_list)


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password') # Plaintext password from request

    if not username or not password:
         return jsonify({'status': 'fail', 'message': 'Username and password required'}), 400

    user = User.query.filter_by(username=username).first()

    # --- << CHANGE: Check hashed password >> ---
    if user and user.check_password(password):
    # ----------------------------------------
        # Password matches
        # Use the to_dict method (ensure it includes has_selected_preferences)
        user_data = user.to_dict()
        # Add Flask-Login logic here if using it: login_user(user)
        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'user': user_data
        })
    else:
        # Password doesn't match or user doesn't exist
        return jsonify({
            'status': 'fail',
            'message': 'Invalid credentials'
        }), 401

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password') # Plaintext password from request

    if not username or not email or not password:
         return jsonify({'status': 'fail', 'message': 'Username, email, and password required'}), 400

    # Check if user with same username or email already exists
    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({
            'status': 'fail',
            'message': 'User with that username or email already exists'
        }), 400

    # Create user object without password first
    new_user = User(username=username, email=email)
    # --- << CHANGE: Set hashed password >> ---
    new_user.set_password(password)
    # -------------------------------------

    db.session.add(new_user)
    try:
        db.session.commit()
        # Use the to_dict method to get user data (without hash)
        user_data = new_user.to_dict()
        return jsonify({
            'status': 'success',
            'message': 'User created successfully',
            'user': user_data
        }), 201
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error during signup commit for {username}: {e}", exc_info=True)
        return jsonify({'status': 'fail', 'message': 'Database error during signup'}), 500
    
# Get a user's liked books
@app.route('/user/<int:user_id>/liked', methods=['GET'])
def get_liked_books(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'fail', 'message': 'User not found'}), 404
    
    books_list = [book.to_dict() for book in user.liked_books]
    return jsonify(books_list)

# Get a user's liked books
@app.route('/user/<int:user_id>/liked', methods=['POST'])
def add_liked_book(user_id):
    data = request.get_json()
    google_book_id = data.get('google_book_id')
    
    # Check if the user exists
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'fail', 'message': 'User not found'}), 404
    
    # Check if the book already exists in our DB; if not, create it
    book = Book.query.filter_by(google_book_id=google_book_id).first()
    if not book:
        # Safely convert rating; if missing or not valid, default to None
        rating_value = data.get('rating')
        try:
            rating_value = float(rating_value) if rating_value not in [None, '', 'N/A'] else None
        except (ValueError, TypeError):
            rating_value = None

        book = Book(
            google_book_id=google_book_id,
            title=data.get('title'),
            authors=data.get('authors'),
            genre=data.get('genre', 'N/A'),  # Added genre with default value
            synopsis=data.get('synopsis'),
            rating=rating_value,
            image_link=data.get('image_link')
        )
        db.session.add(book)
        db.session.commit()
    
    # Check if book is already liked
    if book in user.liked_books:
        return jsonify({'status': 'success', 'message': 'Book already liked'})
    
    # Add to liked books
    user.liked_books.append(book)
    db.session.commit()
    
    return jsonify({
        'status': 'success', 
        'message': 'Book added to liked books',
        'book': book.to_dict()
    })
    
@app.route('/api/book/<string:google_book_id>/sync', methods=['POST'])
def sync_book_details_from_api(google_book_id):
    """
    Fetches latest details from Google Books API and updates/creates
    the corresponding record in the local 'books' table.
    """
    app.logger.info(f"Sync requested for book ID: {google_book_id}")
    try:
        # --- Fetch from Google Books API ---
        response = service.volumes().get(volumeId=google_book_id).execute()
        app.logger.debug(f"Google API response for sync: {response}")

        volume_info = response.get('volumeInfo', {})
        sale_info = response.get('saleInfo', {})
        image_links = volume_info.get('imageLinks', {})

        # --- Process Data (same logic as before) ---
        image_url = (image_links.get('extraLarge') or image_links.get('large') or
                     image_links.get('medium') or image_links.get('thumbnail'))
        categories = volume_info.get('categories', [])
        genre = categories[0] if categories else 'N/A'
        list_price = sale_info.get('listPrice') # Dict or None
        buy_link = sale_info.get('buyLink')     # String or None
        rating_value = volume_info.get('averageRating')
        try: rating = float(rating_value) if rating_value is not None else None
        except (ValueError, TypeError): rating = None
        authors_list = volume_info.get('authors', ['Unknown'])
        authors_str = ', '.join(authors_list) if isinstance(authors_list, list) else 'Unknown'


        # --- Find existing book or create new ---
        book = Book.query.filter_by(google_book_id=google_book_id).first()

        if book:
            # Update existing book
            app.logger.info(f"Updating existing book record for {google_book_id}")
            book.title = volume_info.get('title', book.title) # Keep old if new is missing
            book.authors = authors_str
            book.genre = genre
            book.synopsis = volume_info.get('description', book.synopsis)
            book.rating = rating
            book.image_link = image_url
            # Add price/link if needed in Book model, or store elsewhere
            # book.listPrice = list_price # Requires Book model change
            # book.buyLink = buy_link    # Requires Book model change
        else:
            # Create new book
            app.logger.info(f"Creating new book record for {google_book_id}")
            book = Book(
                google_book_id=google_book_id,
                title=volume_info.get('title', 'N/A'),
                authors=authors_str,
                genre=genre,
                synopsis=volume_info.get('description', 'No synopsis available.'),
                rating=rating,
                image_link=image_url
                # Add price/link if needed in Book model
            )
            db.session.add(book)

        db.session.commit()
        app.logger.info(f"Successfully synced details for book ID: {google_book_id}")
        # Return simple success - data will be fetched via the GET route
        return jsonify({'status': 'success', 'message': 'Book details synced to DB'})

    # --- Error Handling (same as before, adjusted messages) ---
    except HttpError as e:
        status_code = e.resp.status if hasattr(e, 'resp') else 500
        error_message = f"Sync failed: Google Books API error (Status: {status_code}): {str(e)}"
        app.logger.error(f"HttpError syncing {google_book_id}: {error_message}", exc_info=True)
        if status_code == 404: error_message = "Sync failed: Book not found via Google Books API."
        return jsonify({'status': 'fail', 'error': error_message}), status_code
    except requests.exceptions.SSLError as ssl_e:
         error_message = f"Sync failed: SSL Error connecting to Google Books API: {str(ssl_e)}."
         app.logger.error(f"SSLError syncing {google_book_id}: {error_message}", exc_info=True)
         return jsonify({'status': 'fail', 'error': error_message}), 500
    except Exception as e:
        db.session.rollback() # Rollback DB changes on general error
        error_message = f"Sync failed: Unexpected error processing book {google_book_id}: {str(e)}"
        if "[SSL: WRONG_VERSION_NUMBER]" in str(e): error_message += " [SSL Error]"
        app.logger.error(f"Unexpected error syncing {google_book_id}: {error_message}", exc_info=True)
        return jsonify({'status': 'fail', 'error': error_message}), 500


# === Function 2: Get Book Details FROM Local DB ===
@app.route('/api/book/<string:google_book_id>', methods=['GET'])
def get_book_details_from_db(google_book_id):
    """Fetches book details from the local database."""
    app.logger.info(f"Fetching book details from DB for ID: {google_book_id}")
    try:
        # Query only the local 'books' table
        book = Book.query.filter_by(google_book_id=google_book_id).first()

        if book:
            app.logger.info(f"Found book {google_book_id} in local DB.")
            # Use the Book model's to_dict() method
            book_details = book.to_dict()

            # --- Manually add Price/Buy Link if NOT stored on Book model ---
            # If you didn't add price/link columns to the main Book model,
            # you might need to fetch them separately here if required,
            # or ideally, the sync function should have added them.
            # For simplicity, we assume to_dict includes everything needed
            # OR that price/link aren't strictly needed from *this* endpoint anymore
            # if the sync endpoint is always called first by the frontend.
            # Let's assume price/link are needed and WERE added to the Book model & to_dict.
            # If not, you'd fetch from Google API *here* if book found but price missing.
            # --- Example if price/link were NOT added to Book model: ---
            # try:
            #     g_response = service.volumes().get(volumeId=google_book_id, projection='lite', fields='saleInfo(listPrice,buyLink)').execute()
            #     sale_info = g_response.get('saleInfo', {})
            #     book_details['listPrice'] = sale_info.get('listPrice')
            #     book_details['buyLink'] = sale_info.get('buyLink')
            # except Exception as api_err:
            #     app.logger.warn(f"Could not fetch saleInfo separately for {google_book_id}: {api_err}")
            #     book_details['listPrice'] = None
            #     book_details['buyLink'] = None
            # --------------------------------------------------------

            return jsonify(book_details)
        else:
            app.logger.warn(f"Book {google_book_id} not found in local DB.")
            # Inform frontend the book needs to be synced first
            return jsonify({'error': 'Book not found in local database. Please sync first.'}), 404

    except Exception as e:
        error_message = f"Error retrieving book details from database for {google_book_id}: {str(e)}"
        app.logger.error(error_message, exc_info=True)
        return jsonify({'error': 'Database query error'}), 500

    
# Endpoint to remove a book from a user's liked books
@app.route('/user/<int:user_id>/liked/<string:book_id>', methods=['DELETE'])
def remove_liked_book(user_id, book_id):
    # Find the user
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'fail', 'message': 'User not found'}), 404
    
    # Find the book by Google Book ID
    book = Book.query.filter_by(google_book_id=book_id).first()
    if not book:
        return jsonify({'status': 'fail', 'message': 'Book not found'}), 404
    
    # Check if book is in the user's liked books
    if book not in user.liked_books:
        return jsonify({'status': 'fail', 'message': 'Book not liked by user'}), 404
    
    # Remove the book from liked books
    user.liked_books.remove(book)
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': 'Book removed from liked books'})

# Get a user's playlist
@app.route('/user/<int:user_id>/playlist', methods=['GET'])
def get_playlist(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'fail', 'message': 'User not found'}), 404
    
    # Get the playlist with ordered books and include the tag
    playlist_items = db.session.query(Book, playlist_books.c.position, playlist_books.c.tag)\
        .join(playlist_books, Book.id == playlist_books.c.book_id)\
        .filter(playlist_books.c.user_id == user_id)\
        .order_by(playlist_books.c.position).all()
    
    books_list = []
    for book, position, tag in playlist_items:
        book_dict = book.to_dict()
        book_dict['position'] = position
        book_dict['tag'] = tag
        books_list.append(book_dict)
    
    return jsonify(books_list)

@app.route('/user/<int:user_id>/playlist/update_tag', methods=['PUT'])
def update_playlist_tag(user_id):
    data = request.get_json()
    google_book_id = data.get('google_book_id')
    new_tag = data.get('tag')
    
    # Find the book by Google Book ID
    book = Book.query.filter_by(google_book_id=google_book_id).first()
    if not book:
        return jsonify({'status': 'fail', 'message': 'Book not found'}), 404

    # Update the tag in the association table
    stmt = playlist_books.update().\
        where(
            playlist_books.c.user_id == user_id,
            playlist_books.c.book_id == book.id
        ).\
        values(tag=new_tag)
    db.session.execute(stmt)
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': 'Tag updated', 'tag': new_tag})




# Endpoint to add a book to a user's playlist from the genre
@app.route('/user/<int:user_id>/playlist_genre', methods=['POST'])
def add_playlist_book_genre(user_id):
    data = request.get_json()
    google_book_id = data.get('google_book_id')
    tag = data.get('tag', 'save_later')  # Use provided tag or default to 'save_later'
    
    # Check if the user exists
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'fail', 'message': 'User not found'}), 404
    
    # Check if the book already exists in our DB; if not, create it
    book = Book.query.filter_by(google_book_id=google_book_id).first()
    if not book:
        # Safely convert rating; if missing or not valid, default to None
        rating_value = data.get('rating')
        try:
            rating_value = float(rating_value) if rating_value not in [None, '', 'N/A'] else None
        except (ValueError, TypeError):
            rating_value = None

        book = Book(
            google_book_id=google_book_id,
            title=data.get('title'),
            authors=data.get('authors'),
            genre=data.get('genre', 'N/A'),  # Added genre with default value
            synopsis=data.get('synopsis'),
            rating=rating_value,
            image_link=data.get('image_link')
        )
        db.session.add(book)
        db.session.commit()
    
    # Check if book is already in the playlist
    if book in user.playlist_books:
        return jsonify({'status': 'success', 'message': 'Book already in playlist'})
    
    # Get the highest position in the playlist
    max_position_result = db.session.query(db.func.max(playlist_books.c.position))\
        .filter(playlist_books.c.user_id == user_id).first()
    
    next_position = 1  # Default to 1 if no books in playlist
    if max_position_result[0] is not None:
        next_position = max_position_result[0] + 1
    
    # Insert into the association table including the tag field
    stmt = playlist_books.insert().values(
        user_id=user_id,
        book_id=book.id,
        position=next_position,
        tag=tag
    )
    db.session.execute(stmt)
    db.session.commit()
    
    return jsonify({
        'status': 'success', 
        'message': 'Book added to playlist',
        'book': book.to_dict()
    })
    
@app.route('/user/<int:user_id>/generate-recommendations', methods=['POST'])
def trigger_recommendations(user_id):
    """
    Generates and saves recommendations based on the user's current playlist.
    Assumes you have a function:
        generate_and_save_playlist_recommendations(user_id)
    which returns a dict like {'status':'success', 'message':..., ...}
    or {'status':'fail','message':...} on error.
    """
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'fail', 'message': 'User not found'}), 404
    
    result_1 = add_recommendations(user_id,db)

    try:
        result = generate_and_save_playlist_recommendations(user_id,db)
        # Validate that we got a dict back
        if not isinstance(result, dict):
            app.logger.error(
                f"generate_and_save_playlist_recommendations returned invalid type: {type(result)}"
            )
            return jsonify({
                'status': 'fail',
                'message': 'Internal error: invalid recommendation response.'
            }), 500

        # Propagate success or failure from the helper
        if result.get('status') == 'success':
            return jsonify(result)
        else:
            # Use 400 for client‐level errors, or 500 if you prefer
            return jsonify(result), 400

    except Exception as e:
        app.logger.error(
            f"Unexpected error generating recommendations for user {user_id}: {e}",
            exc_info=True
        )
        return jsonify({
            'status': 'fail',
            'message': 'An unexpected server error occurred.'
        }), 500
        
# --- Endpoint 2: Clear Playlist ---
@app.route('/user/<int:user_id>/clear-playlist', methods=['DELETE']) # Using DELETE method is conventional for clearing/deleting resources
def clear_user_playlist(user_id):
    """
    Deletes all books from the specified user's playlist.
    """
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'fail', 'message': 'User not found'}), 404

    try:
        # Create and execute the delete statement for the playlist_books table
        stmt = playlist_books.delete().where(playlist_books.c.user_id == user_id)
        db.session.execute(stmt)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': 'User playlist cleared successfully.'
        })

    except Exception as e:
        db.session.rollback() # Rollback if deletion fails
        print(f"Error clearing playlist for user {user_id}: {e}")
        return jsonify({'status': 'fail', 'message': f'An error occurred while clearing playlist: {e}'}), 500


# Endpoint to add a book to a user's playlist
@app.route('/user/<int:user_id>/playlist', methods=['POST'])
def add_playlist_book(user_id):
    data = request.get_json()
    google_book_id = data.get('google_book_id')
    tag = data.get('tag', 'save_later')  # Use provided tag or default to 'save_later'
    
    # Check if the user exists
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'fail', 'message': 'User not found'}), 404
    
    # Check if the book already exists in our DB; if not, create it
    book = Book.query.filter_by(google_book_id=google_book_id).first()
    if not book:
        # Safely convert rating; if missing or not valid, default to None
        rating_value = data.get('rating')
        try:
            rating_value = float(rating_value) if rating_value not in [None, '', 'N/A'] else None
        except (ValueError, TypeError):
            rating_value = None

        book = Book(
            google_book_id=google_book_id,
            title=data.get('title'),
            authors=data.get('authors'),
            genre=data.get('genre', 'N/A'),  # Added genre with default value
            synopsis=data.get('synopsis'),
            rating=rating_value,
            image_link=data.get('image_link')
        )
        db.session.add(book)
        db.session.commit()
        
    
    # Check if book is already in the playlist
    if book in user.playlist_books:
        return jsonify({'status': 'success', 'message': 'Book already in playlist'})
    
    # Get the highest position in the playlist
    max_position_result = db.session.query(db.func.max(playlist_books.c.position))\
        .filter(playlist_books.c.user_id == user_id).first()
    
    next_position = 1  # Default to 1 if no books in playlist
    if max_position_result[0] is not None:
        next_position = max_position_result[0] + 1
    
    # Insert into the association table including the tag field
    stmt = playlist_books.insert().values(
        user_id=user_id,
        book_id=book.id,
        position=next_position,
        tag=tag
    )
    db.session.execute(stmt)
    db.session.commit()
    
    # Use a unique variable name
    bg_thread_1 = threading.Thread(
        target=add_recommendations,  # Pass function REFERENCE
        args=(user_id,)             # Pass arguments as a TUPLE
    )
    bg_thread_1.daemon = True # Allow app to exit even if thread runs
    bg_thread_1.start()     # Start the first thread

    # --- Start Thread 2: Calling generate_and_save_playlist_recommendations ---
    # Make sure this function is defined and takes only user_id
    # and uses app_context INTERNALLY (as corrected before)
    app.logger.info(f"Starting thread for generate_and_save_playlist_recommendations for user {user_id}")
    # Use a different variable name
    bg_thread_2 = threading.Thread(
        target=generate_and_save_playlist_recommendations, # Pass function REFERENCE
        args=(user_id,)                                     # Pass arguments as a TUPLE
    )
    bg_thread_2.daemon = True
    bg_thread_2.start()     # Start the second thread

    
    return jsonify({
        'status': 'success', 
        'message': 'Book added to playlist',
        'book': book.to_dict()
    })

# Endpoint to remove a book from a user's playlist
@app.route('/user/<int:user_id>/playlist/<string:book_id>', methods=['DELETE'])
def remove_playlist_book(user_id, book_id):
    # Find the user
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'fail', 'message': 'User not found'}), 404
    
    # Find the book by Google Book ID
    book = Book.query.filter_by(google_book_id=book_id).first()
    if not book:
        return jsonify({'status': 'fail', 'message': 'Book not found'}), 404
    
    # Check if book is in the user's playlist
    if book not in user.playlist_books:
        return jsonify({'status': 'fail', 'message': 'Book not in playlist'}), 404
    
    # Get the position of the book to be removed
    position_result = db.session.query(playlist_books.c.position)\
        .filter(playlist_books.c.user_id == user_id, playlist_books.c.book_id == book.id).first()
    
    if position_result:
        removed_position = position_result[0]
        
        # Remove the book from playlist
        stmt = playlist_books.delete().where(
            playlist_books.c.user_id == user_id,
            playlist_books.c.book_id == book.id
        )
        db.session.execute(stmt)
        
        # Reorder the positions of remaining books
        stmt = playlist_books.update()\
            .where(
                playlist_books.c.user_id == user_id,
                playlist_books.c.position > removed_position
            )\
            .values(position=playlist_books.c.position - 1)
        db.session.execute(stmt)
        
        db.session.commit()
    
    return jsonify({'status': 'success', 'message': 'Book removed from playlist'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    #bg_thread = threading.Thread(target=run_background_recommendations_for_all)
    #bg_thread.daemon = True  # Ensures this thread won't block shutdown.
    #bg_thread.start()
    app.run(debug=True, port=5000)