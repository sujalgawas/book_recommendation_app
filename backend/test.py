# server.py (or app.py)

from flask import Flask, jsonify
import torch
import pickle
import random
import pandas as pd  # Import pandas to read the CSV
from model_utils import load_model, CandidateGenerationModel # Assuming this import is correct
import logging

# Device setup
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Using device:", device)

print(torch.__version__) 
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device(),
          "| Name:", torch.cuda.get_device_name(0))

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
    model.eval() # Set model to evaluation mode
    print("Successfully loaded recommendation model.")
except FileNotFoundError:
    print("ERROR: Model file 'models/candidate_model.pt' not found.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    exit()


app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    # --- Select a random user (or get from request args later) ---
    # Ensure user2idx is not empty
    if not user2idx:
        return jsonify({"error": "User mapping is empty."}), 500
    user_id = random.choice(list(user2idx.keys()))
    user_idx_tensor = torch.tensor([user2idx[user_id]], dtype=torch.long).to(device)
    print(f"Generating recommendations for random user: {user_id} (index: {user_idx_tensor.item()})")

    scores_with_indices = []
    # --- Iterate through all known book *indices* ---
    all_book_indices = list(book2idx.values()) # Get all possible internal indices
    if not all_book_indices:
        return jsonify({"error": "Book mapping is empty."}), 500

    # Create a tensor of all book indices for potentially faster batch processing if model supports it
    # If model only supports one book at a time, keep the loop
    all_book_indices_tensor = torch.tensor(all_book_indices, dtype=torch.long).to(device)

    # --- Score books ---
    try:
        with torch.no_grad(): # Disable gradient calculation for inference
            # --- << METHOD 1: Batch Scoring (if model supports broadcasting user_idx) >> ---
            # This is generally much faster if possible
            # scores_tensor = model(user_idx_tensor.expand_as(all_book_indices_tensor), all_book_indices_tensor)
            # scores = scores_tensor.cpu().numpy() # Get scores as numpy array
            # scores_with_indices = list(zip(all_book_indices, scores))
            # -----------------------------------------------------------------------------

            # --- << METHOD 2: Scoring one by one (Fallback if batching doesn't work) >> ---
            for book_idx in all_book_indices:
                book_idx_tensor = torch.tensor([book_idx], dtype=torch.long).to(device)
                score = model(user_idx_tensor, book_idx_tensor).item()
                scores_with_indices.append((book_idx, score)) # Store internal index and score
            # -----------------------------------------------------------------------------


        # --- Sort by score and get top 10 ---
        # Sort based on the score (the second element in the tuple)
        top_results = sorted(scores_with_indices, key=lambda x: x[1], reverse=True)[:10]

        # --- Prepare response ---
        recommendations = []
        for book_index, score in top_results:
            # Map the internal book index back to the original book ID
            original_book_id = idx2book.get(book_index, 'UNKNOWN_ID')
            # Get the title using the original book ID
            title = book_id_to_title.get(str(original_book_id), "Title Not Found")
            recommendations.append({"book_id": original_book_id, "title": title, "score": round(score, 4)})

        print(f"Top 10 recommendations for user {user_id}: {recommendations}")

        # --- Return JSON with book IDs and titles ---
        return jsonify({
            "test_user": user_id, # Keep track of which user was used
            "recommendations": recommendations
        })

    except Exception as e:
        app.logger.error(f"Error during recommendation generation for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate recommendations."}), 500


if __name__ == "__main__":
    # Add basic logging configuration if not done elsewhere
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)
    # Consider host='0.0.0.0' if running in a container or need external access
    app.run(debug=True, port=5000) # Use the same port as before (default is 5000)