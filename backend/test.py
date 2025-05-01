import praw
import os # Optional: For loading credentials from environment variables

# --- Reddit API Credentials ---
# IMPORTANT: Replace these with your actual credentials!
# It's recommended to use environment variables or a config file
# instead of hardcoding them directly in the script for security.
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "p-0U8U54KenxMDZAGyuE2w") # Replace with your client ID
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "u1TyCeDKIEm05YusVWLj-FQr5RdkXw") # Replace with your client secret
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "script:my-book-review-app:v1.0 (by /u/your_username)") # Replace with your user agent

# --- Configuration ---
# Subreddits to search within (add more relevant ones if needed)
DEFAULT_SUBREDDITS = [
    'books',
    'literature',
    'suggestmeabook',
    'booksuggestions',
    'Fantasy',
    'SciFi',
    'freeEbooks',
    '52book',
    'whatsthatbook',
    'bookshelf',
    'Outlander',
    'YAlit',
    'bookexchange',
]

# Maximum number of posts to fetch per subreddit
LIMIT_PER_SUBREDDIT = 10

def get_reddit_reviews(book_title, subreddits=None, limit=LIMIT_PER_SUBREDDIT):
    """
    Searches specified Reddit subreddits for posts discussing a book title.

    Args:
        book_title (str): The title of the book to search for.
        subreddits (list, optional): A list of subreddit names to search within.
                                     Defaults to DEFAULT_SUBREDDITS.
        limit (int, optional): The maximum number of posts to retrieve per subreddit.
                               Defaults to LIMIT_PER_SUBREDDIT.

    Returns:
        list: A list of dictionaries, where each dictionary contains information
              about a relevant Reddit post (title, score, url, subreddit).
              Returns an empty list if an error occurs or no posts are found.
    """
    if subreddits is None:
        subreddits = DEFAULT_SUBREDDITS

    reviews = []

    # Check if credentials are placeholders
    if REDDIT_CLIENT_ID == "YOUR_CLIENT_ID" or REDDIT_CLIENT_SECRET == "YOUR_CLIENT_SECRET":
        print("ERROR: Please replace placeholder Reddit API credentials.")
        return reviews

    try:
        # Initialize PRAW (read-only instance is sufficient for searching)
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
        # Verify read-only status (optional)
        print(f"Reddit API Read-Only Status: {reddit.read_only}")
        print("-" * 20)

        # Construct the search query - adding "review" might help narrow results
        # Using quotes around the title helps find exact matches
        search_query = f'"{book_title}" spoiler free review'
        print(f"Searching for: '{search_query}' in subreddits: {', '.join(subreddits)}")
        print("-" * 20)


        for sub_name in subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)
                print(f"Searching in r/{sub_name}...")

                # Search for submissions (posts) matching the query
                # Sorting by relevance or top might yield better results initially
                search_results = subreddit.search(search_query, sort='relevance', limit=limit)

                found_count = 0
                for submission in search_results:
                    # Basic filtering: Ensure the book title is likely in the post title or selftext
                    # This is a simple check; more advanced NLP could be used.
                    if book_title.lower() in submission.title.lower() or \
                       (submission.selftext and book_title.lower() in submission.selftext.lower()):

                        reviews.append({
                            'title': submission.title,
                            'score': submission.score,
                            'url': submission.url,
                            'subreddit': sub_name,
                            'body_snippet': submission.selftext[:150] + '...' if submission.selftext else '[No body text]' # Add a snippet
                        })
                        found_count += 1

                if found_count == 0:
                    print(f" -> No relevant posts found in r/{sub_name} for this query.")
                else:
                     print(f" -> Found {found_count} potential post(s) in r/{sub_name}.")


            except praw.exceptions.PRAWException as e:
                print(f"Error searching subreddit r/{sub_name}: {e}")
            except Exception as e:
                 print(f"An unexpected error occurred while searching r/{sub_name}: {e}")


    except praw.exceptions.PRAWException as e:
        print(f"ERROR: Failed to connect to Reddit or PRAW error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return reviews

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with the book title you want to search for
    example_book_title = "Project Hail Mary"

    print(f"Attempting to fetch Reddit reviews for: '{example_book_title}'")
    print("=" * 40)

    fetched_reviews = get_reddit_reviews(example_book_title)

    print("=" * 40)
    if fetched_reviews:
        print(f"Found {len(fetched_reviews)} potential review posts:")
        for i, review in enumerate(fetched_reviews, 1):
            print(f"\n--- Post {i} ---")
            print(f"  Subreddit: r/{review['subreddit']}")
            print(f"  Title: {review['title']}")
            print(f"  Score: {review['score']}")
            print(f"  URL: {review['url']}")
            print(f"  Snippet: {review['body_snippet']}")
    else:
        print("No relevant Reddit posts found or an error occurred.")
