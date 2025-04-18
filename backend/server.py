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

app = Flask(__name__)
CORS(app)

# Configure the PostgreSQL database here
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost:5432/book'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

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


# Initialize the Books API service
#service = build('books', 'v1', developerKey='AIzaSyCgtyM44wxtiprQtArM4CIGJk9Ap0wdk-U', cache_discovery=False)
service = build('books', 'v1', developerKey='AIzaSyC3oAIwtxAqCKNtpXFwYUij2OIzSTz1o3s',cache_discovery=False)
    
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
    # Get the user's recommendations sorted by position
    recommendation_items = db.session.query(Book, recommend.c.position)\
        .join(recommend, Book.id == recommend.c.book_id)\
        .filter(recommend.c.user_id == user_id)\
        .order_by(recommend.c.position).all()
    
    books_list = []
    for book, position in recommendation_items:
        # Create a dictionary with the same structure expected by the frontend
        books_list.append({
            'id': book.google_book_id,  # Frontend uses book.id in some places
            'google_book_id': book.google_book_id,  # Used by the frontend for identification
            'title': book.title or 'N/A',
            'authors': book.authors or 'N/A',
            'genre': book.genre or 'N/A',
            'synopsis': book.synopsis or 'N/A',
            'rating': book.rating if book.rating is not None else 'N/A',
            'image_link': book.image_link or 'N/A',
            'position': position,
            'tag': 'recommend'  # Adding tag for frontend identification
        })
    
    # If there are more than 10 recommendations, pick 10 random ones;
    # otherwise, just use the entire list.
    if len(books_list) > 30:
        random_books = random.sample(books_list, 30)
    else:
        random_books = books_list
    
    # Return the result as JSON
    return jsonify(random_books)


def add_recommendations(user_id):
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
    
def run_background_recommendations_for_all():
    with app.app_context():
        try:
            logger.info("Starting background recommendations for all users...")
            # Query all users; ensure your User model is imported.
            users = User.query.all()
            for user in users:
                try:
                    add_recommendations(user.id)
                    logger.info(f"Recommendations added for user_id: {user.id}")
                except Exception as e:
                    logger.error(f"Error processing recommendations for user_id {user.id}: {e}")
            logger.info("Finished background recommendations for all users.")
        except Exception as e:
            logger.error(f"Error in background recommendation process: {e}")



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
    Generates recommendations based on the user's current playlist.
    """
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'fail', 'message': 'User not found'}), 404

    try:
        # Call the function that processes the playlist and adds recommendations
        result = add_recommendations(user_id) # Assuming this function exists

        # Optional: Check result if needed
        # if result.get('status') != 'success':
        #     return jsonify({'status': 'fail', 'message': 'Failed to generate recommendations'}), 500

        return jsonify({
            'status': 'success',
            'message': 'AI recommendations generated successfully based on current playlist.'
        })

    except Exception as e:
        db.session.rollback() # Rollback if recommendation generation fails
        print(f"Error generating recommendations for user {user_id}: {e}")
        return jsonify({'status': 'fail', 'message': f'An error occurred during recommendation generation: {e}'}), 500

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
        
    bg_thread = threading.Thread(target=run_background_recommendations_for_all)
    bg_thread.daemon = True  # Ensures this thread won't block shutdown.
    bg_thread.start()
    
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