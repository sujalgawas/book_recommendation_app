// src/components/BookPage.js

import React, { useState, useEffect, useCallback } from 'react'; // Added useCallback
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Heart, ListPlus, ArrowLeft, Star, BookOpen, CheckSquare, Square, Bookmark, Trash2 } from 'lucide-react';
import axios from 'axios';
// Import CSS
import './BookPage.css';

// Configure axios instance (optional but recommended)
const apiClient = axios.create({
    baseURL: 'http://localhost:5000', // Your Flask backend URL
    withCredentials: true,
});


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

    // --- getUserId ---
    const getUserId = useCallback(() => { // Wrap in useCallback
        try {
            const userString = localStorage.getItem('user');
            if (!userString) return null;
            const user = JSON.parse(userString);
            return user?.id;
        } catch (e) {
            console.error("Error parsing user from localStorage", e);
            return null;
        }
    }, []); // Empty dependency array means it's created once

    // --- Fetch Data Logic ---
    const fetchBookFromDb = useCallback(async (bookId) => {
        console.log(`Attempting to fetch book ${bookId} from local DB...`);
        try {
            // Use GET request to fetch from DB via backend
            const response = await apiClient.get(`/api/book/${bookId}`);
            console.log("Fetched Book Data from DB:", response.data);
            setBook(response.data); // Set book data from local DB
            setError(null); // Clear previous errors
            return true; // Indicate success
        } catch (err) {
            if (err.response && err.response.status === 404) {
                console.log(`Book ${bookId} not found in local DB.`);
                // Book not found, normal case for first view, return false
                return false;
            } else {
                // Other error fetching from DB
                console.error('Error fetching book details from DB:', err.response || err);
                setError(err.response?.data?.error || err.message || 'Failed to fetch book details from database.');
                throw err; // Re-throw other errors to stop the process
            }
        }
    }, []); // Empty dependency array

    const syncAndRefetchBook = useCallback(async (bookId) => {
        console.log(`Book ${bookId} not in DB, triggering API sync...`);
        setIsSyncing(true); // Show syncing indicator
        setError(null);
        try {
            // Call the new POST endpoint to fetch from API and save to DB
            const syncResponse = await apiClient.post(`/api/book/${bookId}/sync`);
            console.log("Sync API Response:", syncResponse.data);
            if (syncResponse.data.status !== 'success') {
                throw new Error(syncResponse.data.error || syncResponse.data.message || 'Failed to sync book details from API.');
            }
            // If sync succeeded, fetch the newly saved data from DB
            await fetchBookFromDb(bookId); // This will set the book state on success

        } catch (err) {
            console.error('Error during sync or re-fetch:', err);
            setError(err.response?.data?.error || err.message || 'Failed to sync and retrieve book details.');
            // Keep book state as null if sync/refetch fails
            setBook(null);
        } finally {
            setIsSyncing(false); // Hide syncing indicator
        }
    }, [fetchBookFromDb]); // Depends on fetchBookFromDb

    // --- Main useEffect for Loading Data ---
    useEffect(() => {
        const loadData = async () => {
            setIsLoading(true); // Start overall loading
            setBook(null); // Reset state
            setError(null);
            setSimilarBooks([]);
            setIsLiked(false);
            setIsInPlaylist(false);
            setCurrentTag('save_later');

            if (!id) {
                setError("No book ID provided in URL.");
                setIsLoading(false);
                return;
            }

            try {
                // 1. Try fetching from local DB first
                const foundInDb = await fetchBookFromDb(id);

                // 2. If not found in DB, trigger sync and re-fetch
                if (!foundInDb) {
                    await syncAndRefetchBook(id);
                    // fetchBookFromDb inside syncAndRefetchBook handles setting the state
                }
                // If foundInDb was true, book state is already set by fetchBookFromDb

                // 3. Fetch user-specific data and similar books only AFTER main book data is potentially loaded
                // We need to access the 'book' state here, which might have just been set
                // Use a slight delay or check if book state is set before proceeding? Let's check state.

            } catch (err) {
                // Errors from fetchBookFromDb (other than 404) or syncAndRefetchBook are caught here
                console.error("Main useEffect error:", err);
                // Error state is already set within the functions
            } finally {
                setIsLoading(false); // End overall loading
            }
        };

        loadData();
    }, [id, fetchBookFromDb, syncAndRefetchBook]); // Dependencies

    // --- useEffect for Secondary Data (Similar Books, User Data) ---
    // Runs AFTER the main book data is loaded into the 'book' state
    useEffect(() => {
        const fetchSecondaryData = async () => {
            if (!book || !book.google_book_id) return; // Only run if book data is available

            const userId = getUserId();
            const bookId = book.google_book_id;

            // Fetch similar books
            if (book.genre && book.genre !== 'N/A') {
                try {
                    console.log(`Workspaceing similar books for genre: ${book.genre}`);
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

            // Fetch user-specific data
            if (userId) {
                console.log(`Workspaceing user (${userId}) data for book ${bookId}...`);
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
                            setCurrentTag('save_later'); // Reset tag if not in playlist
                        }
                    }
                } catch (playlistErr) { console.warn("Could not fetch playlist status", playlistErr); }
            }
        };

        fetchSecondaryData();
    }, [book, getUserId]); // Run this effect when the main 'book' state changes


    // --- Action Handlers (toggleLike, updatePlaylistTag, togglePlaylist - Adapt to use apiClient) ---
    const toggleLike = useCallback(async () => { // Wrapped in useCallback
        const userId = getUserId();
        if (!userId || !book?.google_book_id) return;
        const bookId = book.google_book_id;
        const method = isLiked ? 'DELETE' : 'POST';
        const url = isLiked ? `/user/<span class="math-inline">\{userId\}/liked/</span>{bookId}` : `/user/${userId}/liked`; // Adjust URL for DELETE
        const originalIsLiked = isLiked; // For rollback

        setIsLiked(!isLiked); // Optimistic Update

        try {
            const options = { method: method };
            if (method === 'POST') {
                options.data = book; // Send full book data if needed by backend to create entry
                // apiClient automatically sets Content-Type for objects
            }
            await apiClient(url, options); // Use apiClient
            console.log("Like toggled successfully");
        } catch (error) {
            setIsLiked(originalIsLiked); // Rollback
            console.error('Error toggling like:', error.response || error);
            alert(`Failed to ${originalIsLiked ? 'unlike' : 'like'} book: ${error.response?.data?.message || error.message}`);
        }
    }, [isLiked, book, getUserId]); // Dependencies

    const updatePlaylistTag = useCallback(async (newTag) => { // Wrapped in useCallback
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
    }, [currentTag, book, getUserId]); // Dependencies

    const togglePlaylist = useCallback(async () => { // Wrapped in useCallback
        const userId = getUserId();
        if (!userId || !book?.google_book_id) return;
        const bookId = book.google_book_id;
        const method = isInPlaylist ? 'DELETE' : 'POST';
        const url = isInPlaylist ? `/user/<span class="math-inline">\{userId\}/playlist/</span>{bookId}` : `/user/${userId}/playlist`;
        const originalIsInPlaylist = isInPlaylist;
        const originalTag = currentTag;

        // Optimistic Update
        setIsInPlaylist(!isInPlaylist);
        if (!isInPlaylist) setCurrentTag('save_later');

        try {
            const options = { method: method };
            if (method === 'POST') {
                options.data = { ...book, tag: 'save_later' }; // Send book data for POST
            }
            await apiClient(url, options); // Use apiClient
            console.log(`Successfully toggled playlist`);
        } catch (error) {
            // Rollback
            setIsInPlaylist(originalIsInPlaylist);
            if (!originalIsInPlaylist) setCurrentTag(originalTag);
            console.error('Error toggling playlist:', error.response || error);
            alert(`Failed to toggle playlist: ${error.response?.data?.message || error.message}`);
        }
    }, [isInPlaylist, currentTag, book, getUserId]); // Dependencies


    // --- Render Logic ---

    // Combined Loading State
    if (isLoading || isSyncing) {
        return (
            <div className="book-page loading-container">
                <div className="loading-indicator">
                    {isSyncing ? 'Syncing latest book details...' : 'Loading...'}
                </div>
            </div>
        );
    }

    if (error) { // Display error message
        return (
            <div className="book-page error-container">
                <h2 className="error-message">Error: {error}</h2>
                <button className="home-button" onClick={() => navigate('/')}> Go Home </button>
            </div>
        );
    }

    if (!book) { // Handle case where book is null after loading/syncing
        return (
            <div className="book-page not-found-container">
                <h2 className="not-found-message">Book information could not be loaded.</h2>
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
                        <img src={book.image_link || 'https://via.placeholder.com/300x450.png?text=No+Image'} alt={`${book.title || 'Book'} cover`} className="book-image" />
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
                        <div className="rating-container"> <Star fill={book.rating ? "#FFD700" : "none"} stroke={book.rating ? "#FFD700" : "#ccc"} /> <span style={{ marginLeft: '5px' }}> {typeof book.rating === 'number' ? `${book.rating.toFixed(1)} / 5` : 'No rating'} </span> </div>
                        {book.genre && book.genre !== 'N/A' && (<p className="genre-text">Genre: {book.genre}</p>)}

                        {book.synopsis && book.synopsis !== 'N/A' && (<div> <h2 className="section-title">Synopsis</h2> <p className="synopsis-text"> {book.synopsis} </p> </div>)}
                        {similarBooks.length > 0 && (
                            <div>
                                <h2 className="section-title">Similar Books</h2>
                                <div className="similar-books-grid">
                                    {similarBooks.map((similarBook) => (
                                        <div key={similarBook.google_book_id} className="similar-book-card" onClick={() => navigate(`/book/${similarBook.google_book_id}`)} title={similarBook.title} >
                                            <img src={similarBook.image_link || 'https://via.placeholder.com/150x220.png?text=No+Image'} alt={similarBook.title} className="similar-book-image" />
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