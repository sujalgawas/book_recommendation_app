import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams, useNavigate, Link } from 'react-router-dom';
import { Star } from 'lucide-react'; // Import Star icon for rating
// Removed unused icons: PlusCircle, Heart, BookOpen, CheckSquare, Square, Bookmark, Trash2
// import './SearchResultsPage.css'; // Optional: Add custom styles if needed

const SearchResultsPage = () => {
  const [searchParams] = useSearchParams();
  // Removed useNavigate as it's not used after removing actions
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Removed state for playlist and liked books
  // const [playlist, setPlaylist] = useState([]);
  // const [likedBooks, setLikedBooks] = useState([]);

  const query = searchParams.get('query');

  // Removed getUserId as it's no longer needed
  // const getUserId = useCallback(() => { ... }, []);

  // --- Fetch Search Results (no change) ---
  useEffect(() => {
    if (!query) {
      setResults([]);
      setIsLoading(false);
      return;
    }
    const fetchResults = async () => {
      setIsLoading(true);
      setError(null);
      setResults([]); // Clear previous results
      try {
        // Ensure the backend /search endpoint returns 'rating' in the book object
        const response = await fetch(`http://localhost:5000/search?query=${encodeURIComponent(query)}&num_results=20`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        // Make sure the rating is included in the data from the backend
        setResults(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error("Error fetching search results:", err);
        setError(err.message || "Failed to fetch results.");
        setResults([]);
      } finally {
        setIsLoading(false);
      }
    };
    fetchResults();
  }, [query]);

  // --- Removed useEffect for fetching Playlist and Liked Books ---

  // --- Removed Helper Functions: isBookInPlaylist, isBookLiked ---

  // --- Removed Action Handlers: addToPlaylist, toggleLike, updatePlaylistTag, removeFromPlaylist ---


  // --- Render Logic ---
  return (
    <div className="bg-dark text-light min-vh-100 p-4">
      <div className="container">
        <h1 className="mb-4 h2">Search Results for: "{query || '...'}"</h1>

        {/* Loading, Error, No Results states... */}
        {isLoading && <div className="d-flex justify-content-center py-5"><div className="spinner-border text-light" role="status"><span className="visually-hidden">Loading...</span></div></div>}
        {error && <div className="alert alert-danger">{error}</div>}
        {!isLoading && !error && results.length === 0 && query && <p className="text-muted text-center mt-5">No books found matching your search.</p>}
        {!isLoading && !error && !query && <p className="text-muted text-center mt-5">Please enter a search term in the navbar.</p>}

        {/* Results Grid */}
        {!isLoading && !error && results.length > 0 && (
          <div className="row row-cols-1 row-cols-sm-2 row-cols-md-3 row-cols-lg-5 g-4">
            {results.map((book) => {
              // Use google_book_id primarily, fallback to id
              let bookId = book.google_book_id || book.id;
              if (!bookId) {
                  console.warn('Book missing ID:', book);
                  // Generate a temporary key if ID is truly missing for rendering purposes
                  bookId = `missing-id-${Math.random()}`;
              }
              // Removed liked, playlistEntry, inPlaylist, currentTag variables

              // Removed smallButtonStyle and smallIconSize variables

              return (
                <div className="col" key={bookId}>
                  {/* Use minHeight to keep cards somewhat uniform if content varies */}
                  <div className="card bg-secondary text-light shadow h-100"> {/* Added h-100 for uniform height */}
                    <Link to={bookId.startsWith('missing-id-') ? '#' : `/book/${bookId}`} className="text-decoration-none">
                      <img
                        src={book.image_link || 'https://via.placeholder.com/150x220.png?text=No+Image'}
                        className="card-img-top"
                        alt={book.title || 'Book cover'}
                        style={{ aspectRatio: '2 / 3', objectFit: 'cover' }}
                        onError={(e) => { e.target.onerror = null; e.target.src='https://via.placeholder.com/150x220.png?text=No+Image'; }}
                      />
                    </Link>
                    {/* Adjusted padding and flex for better content layout */}
                    <div className="card-body p-2 d-flex flex-column">
                      <Link to={bookId.startsWith('missing-id-') ? '#' : `/book/${bookId}`} className="text-decoration-none text-light" style={{ flexGrow: 1 }}>
                        <h6 className="card-title small text-light mb-1 text-truncate" title={book.title || 'No Title'}>
                          {book.title || 'No Title'}
                        </h6>
                        <p className="card-text small text-white-50 mb-2 text-truncate" title={Array.isArray(book.authors) ? book.authors.join(', ') : book.authors || 'Unknown Author'}>
                          {Array.isArray(book.authors) ? book.authors.join(', ') : book.authors || 'Unknown Author'}
                        </p>
                      </Link>

                      {/* --- Rating Display --- */}
                      <div className="mt-auto pt-1"> {/* Ensures rating is at the bottom */}
                        <p className={`card-text small mb-0 d-flex align-items-center gap-1 ${book.rating ? 'text-warning' : 'text-muted'}`}>
                          <Star size={14} fill={book.rating ? 'currentColor' : 'none'} />
                          {/* Display rating if available, otherwise 'N/A'. Assumes rating is out of 5 */}
                          <span>
                            {typeof book.rating === 'number' ? `${book.rating.toFixed(1)} / 5` : 'N/A'}
                          </span>
                        </p>
                      </div>
                      {/* --- Removed Action Buttons Area --- */}

                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchResultsPage;