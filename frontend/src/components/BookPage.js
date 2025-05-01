// src/components/BookPage.js

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Heart, ListPlus, ArrowLeft, Star, BookOpen, CheckSquare, Square, Bookmark, Trash2, MessageSquare, ExternalLink, Loader2, AlertTriangle } from 'lucide-react'; // Added icons
import axios from 'axios';
import './BookPage.css'; // Make sure to add styles for Reddit section

// Configure axios instance
const apiClient = axios.create({
    baseURL: 'http://localhost:5000', // Your Flask backend URL
    withCredentials: true,
});

// Helper function to format Reddit timestamps
const formatTimestamp = (utcTimestamp) => {
    if (!utcTimestamp) return 'Unknown date';
    const date = new Date(utcTimestamp * 1000); // Convert seconds to milliseconds
    return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
};


const BookPage = () => {
    const { id } = useParams(); // google_book_id from URL
    const navigate = useNavigate();
    const [book, setBook] = useState(null);
    const [similarBooks, setSimilarBooks] = useState([]);
    const [isLiked, setIsLiked] = useState(false);
    const [isInPlaylist, setIsInPlaylist] = useState(false);
    const [currentTag, setCurrentTag] = useState('save_later');
    const [isLoading, setIsLoading] = useState(true); // Main loading state
    const [isSyncing, setIsSyncing] = useState(false); // Specific state for sync operation
    const [error, setError] = useState(null);

    // --- State for Reddit Reviews ---
    const [redditReviews, setRedditReviews] = useState([]);
    const [isLoadingReddit, setIsLoadingReddit] = useState(false);
    const [redditError, setRedditError] = useState(null);
    // -------------------------------

    const getUserId = useCallback(() => {
        try {
            const userString = localStorage.getItem('user');
            if (!userString) return null;
            const user = JSON.parse(userString);
            return user?.id;
        } catch (e) {
            console.error("Error parsing user from localStorage", e);
            return null;
        }
    }, []);

    const fetchBookFromDb = useCallback(async (bookId) => {
        console.log(`Attempting to fetch book ${bookId} from local DB...`);
        try {
            const response = await apiClient.get(`/api/book/${bookId}`);
            console.log("Fetched Book Data from DB:", response.data);
            setBook(response.data);
            setError(null);
            return true;
        } catch (err) {
            if (err.response && err.response.status === 404) {
                console.log(`Book ${bookId} not found in local DB.`);
                return false;
            } else {
                console.error('Error fetching book details from DB:', err.response || err);
                setError(err.response?.data?.error || err.message || 'Failed to fetch book details from database.');
                throw err;
            }
        }
    }, []);

    const syncAndRefetchBook = useCallback(async (bookId) => {
        console.log(`Book ${bookId} not in DB, triggering API sync...`);
        setIsSyncing(true);
        setError(null);
        try {
            const syncResponse = await apiClient.post(`/api/book/${bookId}/sync`);
            console.log("Sync API Response:", syncResponse.data);
            if (syncResponse.data.status !== 'success' && !syncResponse.data.book) { // Check if sync failed AND didn't return book data
                 throw new Error(syncResponse.data.error || syncResponse.data.message || 'Failed to sync book details from API.');
            }
             // If sync succeeded OR returned book data directly, fetch/set it
            if (syncResponse.data.book) {
                console.log("Sync returned book data directly, using it.");
                setBook(syncResponse.data.book); // Use data returned by sync if available
                setError(null); // Clear error if sync returned data
            } else {
                // If sync didn't return data directly, refetch from DB
                await fetchBookFromDb(bookId);
            }

        } catch (err) {
            console.error('Error during sync or re-fetch:', err);
            setError(err.response?.data?.error || err.message || 'Failed to sync and retrieve book details.');
            setBook(null);
        } finally {
            setIsSyncing(false);
        }
    }, [fetchBookFromDb]);

    // --- Function to fetch Reddit Reviews ---
    const fetchRedditReviews = useCallback(async (title) => {
        if (!title) return; // Don't fetch if title is missing

        console.log(`Fetching Reddit reviews for: ${title}`);
        setIsLoadingReddit(true);
        setRedditError(null);
        setRedditReviews([]); // Clear previous reviews

        try {
            // Encode the title for the URL query parameter
            const encodedTitle = encodeURIComponent(title);
            const response = await apiClient.get(`/api/reddit-reviews?title=${encodedTitle}`);
            console.log("Reddit API Response:", response.data);
            setRedditReviews(Array.isArray(response.data) ? response.data : []);
        } catch (err) {
            console.error('Error fetching Reddit reviews:', err.response || err);
            setRedditError(err.response?.data?.error || err.message || 'Failed to load Reddit discussions.');
            setRedditReviews([]); // Ensure it's an empty array on error
        } finally {
            setIsLoadingReddit(false);
        }
    }, []); // No dependencies needed as title is passed directly

    // --- Main useEffect for Loading Data ---
    useEffect(() => {
        const loadData = async () => {
            setIsLoading(true);
            setBook(null);
            setError(null);
            setSimilarBooks([]);
            setIsLiked(false);
            setIsInPlaylist(false);
            setCurrentTag('save_later');
            // Reset Reddit state too
            setRedditReviews([]);
            setRedditError(null);
            setIsLoadingReddit(false);


            if (!id) {
                setError("No book ID provided in URL.");
                setIsLoading(false);
                return;
            }

            try {
                const foundInDb = await fetchBookFromDb(id);
                if (!foundInDb) {
                    await syncAndRefetchBook(id);
                    // fetchBookFromDb inside syncAndRefetchBook handles setting the state
                    // OR syncAndRefetchBook sets state directly if sync returns data
                }
                 // Note: User data and similar books are fetched in the next effect
                 // Reddit reviews will also be fetched in the next effect based on the book title

            } catch (err) {
                console.error("Main useEffect error:", err);
                // Error state is set within the functions
            } finally {
                setIsLoading(false);
            }
        };
        loadData();
    }, [id, fetchBookFromDb, syncAndRefetchBook]);

    // --- useEffect for Secondary Data (Similar Books, User Data, REDDIT REVIEWS) ---
    useEffect(() => {
        const fetchSecondaryData = async () => {
            if (!book || !book.google_book_id) return; // Only run if book data is available

            const userId = getUserId();
            const bookId = book.google_book_id;
            const bookTitle = book.title; // Get title for Reddit fetch

            // Fetch similar books (existing logic)
            if (book.genre && book.genre !== 'N/A') {
                try {
                    console.log(`Fetching similar books for genre: ${book.genre}`);
                    const similarResponse = await apiClient.get(`/search?query=subject:${encodeURIComponent(book.genre)}&num_results=5`);
                    if (similarResponse.data) {
                        setSimilarBooks(
                            Array.isArray(similarResponse.data)
                                ? similarResponse.data.filter((b) => b.google_book_id !== bookId).slice(0, 4)
                                : []
                        );
                    }
                } catch (similarErr) { console.warn("Could not fetch similar books:", similarErr); }
            }

            // Fetch user-specific data (existing logic)
            if (userId) {
                console.log(`Fetching user (${userId}) data for book ${bookId}...`);
                try { // Liked Status
                    const likedResponse = await apiClient.get(`/user/${userId}/liked`);
                    if (likedResponse.data) {
                        setIsLiked(Array.isArray(likedResponse.data) && likedResponse.data.some((b) => b.google_book_id === bookId));
                    }
                } catch (likedErr) { console.warn("Could not fetch liked status", likedErr); }

                try { // Playlist Status
                    const playlistResponse = await apiClient.get(`/user/${userId}/playlist`);
                    if (playlistResponse.data) {
                        const playlistBook = Array.isArray(playlistResponse.data) ? playlistResponse.data.find((b) => b.google_book_id === bookId) : null;
                        if (playlistBook) {
                            setIsInPlaylist(true);
                            setCurrentTag(playlistBook.tag || 'save_later');
                        } else {
                            setIsInPlaylist(false);
                            setCurrentTag('save_later');
                        }
                    }
                } catch (playlistErr) { console.warn("Could not fetch playlist status", playlistErr); }
            }

            // --- Fetch Reddit Reviews ---
            if (bookTitle) {
                fetchRedditReviews(bookTitle); // Call the function to fetch reddit reviews
            }
            // --------------------------
        };

        fetchSecondaryData();
    // Add fetchRedditReviews to dependency array
    }, [book, getUserId, fetchRedditReviews]);


    // --- Action Handlers (toggleLike, updatePlaylistTag, togglePlaylist - No changes needed here) ---
     const toggleLike = useCallback(async () => {
        const userId = getUserId();
        if (!userId || !book?.google_book_id) return;
        const bookId = book.google_book_id;
        const method = isLiked ? 'DELETE' : 'POST';
        const url = isLiked ? `/user/${userId}/liked/${bookId}` : `/user/${userId}/liked`;
        const originalIsLiked = isLiked;

        setIsLiked(!isLiked); // Optimistic Update

        try {
            const options = { method: method };
            if (method === 'POST') {
                options.data = book;
            }
            await apiClient(url, options);
            console.log("Like toggled successfully");
        } catch (error) {
            setIsLiked(originalIsLiked); // Rollback
            console.error('Error toggling like:', error.response || error);
            alert(`Failed to ${originalIsLiked ? 'unlike' : 'like'} book: ${error.response?.data?.message || error.message}`);
        }
    }, [isLiked, book, getUserId]);

    const updatePlaylistTag = useCallback(async (newTag) => {
        const userId = getUserId();
        if (!userId || !book?.google_book_id) return;
        const bookId = book.google_book_id;
        const oldTag = currentTag;

        setCurrentTag(newTag); // Optimistic Update

        try {
            await apiClient.put(`/user/${userId}/playlist/update_tag`, {
                google_book_id: bookId, tag: newTag
            });
            console.log("Tag updated successfully");
        } catch (error) {
            setCurrentTag(oldTag); // Rollback
            console.error('Error updating tag:', error.response || error);
            alert(`Failed to update tag: ${error.response?.data?.message || error.message}`);
        }
    }, [currentTag, book, getUserId]);

    const togglePlaylist = useCallback(async () => {
        const userId = getUserId();
        if (!userId || !book?.google_book_id) return;
        const bookId = book.google_book_id;
        const method = isInPlaylist ? 'DELETE' : 'POST';
        const url = isInPlaylist ? `/user/${userId}/playlist/${bookId}` : `/user/${userId}/playlist`;
        const originalIsInPlaylist = isInPlaylist;
        const originalTag = currentTag;

        setIsInPlaylist(!isInPlaylist);
        if (!isInPlaylist) setCurrentTag('save_later'); // Reset tag when removing

        try {
            const options = { method: method };
            if (method === 'POST') {
                options.data = { ...book, tag: 'save_later' };
            }
            await apiClient(url, options);
            console.log(`Successfully toggled playlist`);
             // If removing, reset tag state explicitly AFTER successful API call
            if (method === 'DELETE') {
                 setCurrentTag('save_later');
            }
        } catch (error) {
            setIsInPlaylist(originalIsInPlaylist);
            setCurrentTag(originalTag); // Rollback tag as well
            console.error('Error toggling playlist:', error.response || error);
            alert(`Failed to toggle playlist: ${error.response?.data?.message || error.message}`);
        }
    }, [isInPlaylist, currentTag, book, getUserId]);


    // --- Render Logic ---

    if (isLoading || isSyncing) {
        return (
            <div className="book-page loading-container">
                <div className="loading-indicator">
                    <Loader2 className="animate-spin" size={24} />
                    {isSyncing ? 'Syncing latest book details...' : 'Loading book details...'}
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="book-page error-container">
                 <AlertTriangle size={40} className="error-icon" />
                <h2 className="error-message">Error Loading Book</h2>
                <p className="error-details">{error}</p>
                <button className="home-button" onClick={() => navigate('/')}> Go Home </button>
            </div>
        );
    }

    if (!book) {
        return (
            <div className="book-page not-found-container">
                 <AlertTriangle size={40} className="error-icon" />
                <h2 className="not-found-message">Book information could not be loaded.</h2>
                <p className="error-details">The book might not exist or there was an issue retrieving its details.</p>
                <button className="home-button" onClick={() => navigate('/')}> Go Home </button>
            </div>
        );
    }

    // --- Main Page Content ---
    return (
        <div className="book-page">
            <div className="content-wrapper">
                <button className="back-button" onClick={() => navigate(-1)}>
                    <ArrowLeft size={18} /> Back
                </button>
                <div className="book-main-layout">
                    {/* Column 1: Image & Actions */}
                    <div className="column-image-actions">
                        <img
                            src={book.image_link || 'https://placehold.co/300x450/eeeeee/cccccc?text=No+Image'}
                            alt={`${book.title || 'Book'} cover`}
                            className="book-image"
                            onError={(e) => { e.target.onerror = null; e.target.src='https://placehold.co/300x450/eeeeee/cccccc?text=No+Image'; }} // Placeholder on error
                         />
                        <div className="actions-container">
                            <button onClick={toggleLike} className={`action-button ${isLiked ? 'liked' : ''}`} > <Heart fill={isLiked ? 'currentColor' : 'none'} size={18} /> {isLiked ? 'Unlike' : 'Like'} </button>
                            {!isInPlaylist && (<button onClick={togglePlaylist} className="action-button add-playlist" > <ListPlus size={18} /> Add to Playlist </button>)}
                        </div>
                        {isInPlaylist && (
                            <div className="playlist-status-box">
                                <h5>In Your Playlist:</h5>
                                <div className="playlist-buttons-container">
                                    <button onClick={() => updatePlaylistTag('reading')} className={`playlist-tag-button tag-reading ${currentTag === 'reading' ? 'active' : ''}`}> <BookOpen size={16} /> Reading </button>
                                    <button onClick={() => updatePlaylistTag('completed')} className={`playlist-tag-button tag-completed ${currentTag === 'completed' ? 'active' : ''}`}> <CheckSquare size={16} /> Completed </button>
                                    <button onClick={() => updatePlaylistTag('dropped')} className={`playlist-tag-button tag-dropped ${currentTag === 'dropped' ? 'active' : ''}`}> <Square size={16} /> Dropped </button>
                                    <button onClick={() => updatePlaylistTag('save_later')} className={`playlist-tag-button tag-save_later ${currentTag === 'save_later' ? 'active' : ''}`}> <Bookmark size={16} /> Plan to Read </button>
                                    <button onClick={togglePlaylist} title="Remove from Playlist" className="playlist-tag-button remove-button"> <Trash2 size={16} /> Remove </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Column 2: Details */}
                    <div className="column-details">
                        <h1 className="book-title">{book.title || 'Title Not Available'}</h1>
                        <p className="book-authors">by {book.authors || 'Unknown Author'}</p>
                        <div className="rating-container">
                             <Star fill={book.rating ? "#FFD700" : "none"} stroke={book.rating ? "#FFD700" : "#ccc"} size={20}/>
                             <span className="rating-text">
                                {typeof book.rating === 'number' ? `${book.rating.toFixed(1)} / 5` : 'No Google rating'}
                                {book.ratings_count ? ` (${book.ratings_count} ratings)` : ''}
                             </span>
                        </div>
                        {book.genre && book.genre !== 'N/A' && (<p className="genre-text">Genre: {book.genre}</p>)}

                        {book.synopsis && book.synopsis !== 'N/A' && (
                            <div className="synopsis-section">
                                <h2 className="section-title">Synopsis</h2>
                                <p className="synopsis-text"> {book.synopsis} </p>
                            </div>
                        )}

                        {/* --- Reddit Reviews Section --- */}
                        <div className="reddit-reviews-section">
                            <h2 className="section-title">Reddit Discussions</h2>
                            {isLoadingReddit && (
                                <div className="loading-indicator small-loader">
                                     <Loader2 className="animate-spin" size={20} /> Loading Reddit posts...
                                 </div>
                            )}
                            {redditError && (
                                 <div className="error-message small-error">
                                     <AlertTriangle size={18} /> {redditError}
                                 </div>
                            )}
                            {!isLoadingReddit && !redditError && redditReviews.length === 0 && (
                                <p className="no-results-message">No relevant discussions found on Reddit.</p>
                            )}
                            {!isLoadingReddit && !redditError && redditReviews.length > 0 && (
                                <ul className="reddit-reviews-list">
                                    {redditReviews.map((review, index) => (
                                        <li key={index} className="reddit-review-item">
                                            <div className="review-header">
                                                <span className="review-subreddit">r/{review.subreddit}</span>
                                                <span className="review-score">Score: {review.score}</span>
                                                <span className="review-date">{formatTimestamp(review.created_utc)}</span>
                                            </div>
                                            <a href={review.url} target="_blank" rel="noopener noreferrer" className="review-title-link">
                                                {review.title} <ExternalLink size={14} className="external-link-icon"/>
                                            </a>
                                            <p className="review-snippet">{review.body_snippet}</p>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>
                        {/* ----------------------------- */}


                        {similarBooks.length > 0 && (
                            <div className="similar-books-section">
                                <h2 className="section-title">Similar Books</h2>
                                <div className="similar-books-grid">
                                    {similarBooks.map((similarBook) => (
                                        <div key={similarBook.google_book_id} className="similar-book-card" onClick={() => navigate(`/book/${similarBook.google_book_id}`)} title={similarBook.title} >
                                            <img
                                                src={similarBook.image_link || 'https://placehold.co/150x220/eeeeee/cccccc?text=No+Image'}
                                                alt={similarBook.title}
                                                className="similar-book-image"
                                                onError={(e) => { e.target.onerror = null; e.target.src='https://placehold.co/150x220/eeeeee/cccccc?text=No+Image'; }}
                                            />
                                            <p className="similar-book-title">{similarBook.title}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BookPage;
