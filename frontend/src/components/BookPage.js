// src/components/BookPage.js

import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom'; // Added Link
import { Heart, ListPlus, ArrowLeft, Star, BookOpen, CheckSquare, Square, Bookmark, Trash2, ShoppingCart, Tag as PriceTag } from 'lucide-react'; // Added ShoppingCart, PriceTag

// Import the CSS file (make sure it exists and is linked correctly)
import './BookPage.css';

const BookPage = () => {
    const { id } = useParams(); // google_book_id from URL
    const navigate = useNavigate();
    const [book, setBook] = useState(null);
    const [similarBooks, setSimilarBooks] = useState([]);
    const [isLiked, setIsLiked] = useState(false);
    const [isInPlaylist, setIsInPlaylist] = useState(false);
    const [currentTag, setCurrentTag] = useState('save_later');
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null); // Added error state

    // --- getUserId ---
    const getUserId = () => {
        try { // Added try-catch
            const userString = localStorage.getItem('user');
            if (!userString) return null;
            const user = JSON.parse(userString);
            return user?.id; // Return null if no id
         } catch (e) {
            console.error("Error parsing user from localStorage", e);
            return null;
         }
    };

    // --- Fetch Data ---
    useEffect(() => {
        const fetchData = async () => {
            setIsLoading(true);
            setError(null); // Reset error on new fetch
            setBook(null); // Reset book data
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
                // Fetch main book details (including price/link)
                const bookResponse = await fetch(`http://localhost:5000/book/${id}`);
                if (!bookResponse.ok) {
                     const errorData = await bookResponse.json().catch(() => ({})); // Try get error message
                     throw new Error(errorData.error || `Failed to fetch book details (${bookResponse.status})`);
                }
                const bookData = await bookResponse.json();
                 console.log("Fetched Book Data:", bookData); // Log fetched data
                setBook(bookData);

                // Fetch similar books (optional, based on genre)
                if (bookData.genre && bookData.genre !== 'N/A') {
                    try {
                         const similarResponse = await fetch(`http://localhost:5000/search?query=subject:${encodeURIComponent(bookData.genre)}&num_results=5`);
                         if(similarResponse.ok) {
                             const similarData = await similarResponse.json();
                             setSimilarBooks(
                                 Array.isArray(similarData)
                                 ? similarData.filter((b) => b.google_book_id !== id).slice(0, 4)
                                 : []
                             );
                         }
                    } catch (similarErr) {
                        console.warn("Could not fetch similar books:", similarErr); // Non-critical error
                    }
                }

                // Check user-specific data (like/playlist status)
                const userId = getUserId();
                if (userId && bookData.google_book_id) { // Check if book ID exists before fetching user data
                    // Check Liked Status
                    try {
                        const likedResponse = await fetch(`http://localhost:5000/user/${userId}/liked`);
                        if (likedResponse.ok) {
                            const likedData = await likedResponse.json();
                            setIsLiked(Array.isArray(likedData) && likedData.some((b) => b.google_book_id === bookData.google_book_id));
                        }
                    } catch (likedErr) { console.warn("Could not fetch liked status", likedErr); }

                    // Check Playlist Status
                     try {
                         const playlistResponse = await fetch(`http://localhost:5000/user/${userId}/playlist`);
                         if (playlistResponse.ok) {
                             const playlistData = await playlistResponse.json();
                             const playlistBook = Array.isArray(playlistData) ? playlistData.find((b) => b.google_book_id === bookData.google_book_id) : null;
                             if (playlistBook) {
                                 setIsInPlaylist(true);
                                 setCurrentTag(playlistBook.tag || 'save_later');
                             } else {
                                 setIsInPlaylist(false);
                             }
                         }
                     } catch (playlistErr) { console.warn("Could not fetch playlist status", playlistErr); }
                }

            } catch (error) {
                console.error('Error fetching data:', error);
                setError(error.message); // Set error state
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();
    }, [id]); // Removed navigate from dependency array as it's stable

    // --- Action Handlers (toggleLike, updatePlaylistTag, togglePlaylist - Keep as they were) ---
     const toggleLike = async () => {
        const userId = getUserId();
        if (!userId || !book?.google_book_id) return; // Check book id too
        const bookId = book.google_book_id;
        const url = `http://localhost:5000/user/${userId}/liked`;
        const method = isLiked ? 'DELETE' : 'POST';
        const finalUrl = isLiked ? `${url}/${bookId}` : url;
        try {
            const options = {
                 method: method,
                 headers: method === 'POST' ? { 'Content-Type': 'application/json' } : {},
                 // Send full book details on POST in case backend needs to create it
                 body: method === 'POST' ? JSON.stringify(book) : null,
            };
            const response = await fetch(finalUrl, options);
             if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.message || `Failed to ${isLiked ? 'unlike' : 'like'} book`);
             }
            setIsLiked(!isLiked); // Toggle state on success
        } catch (error) {
            console.error('Error toggling like:', error);
            alert(`Failed to ${isLiked ? 'unlike' : 'like'} book. Please try again.`);
        }
    };

    const updatePlaylistTag = async (newTag) => {
        const userId = getUserId();
        if (!userId || !book?.google_book_id) return;
         const bookId = book.google_book_id;
         const oldTag = currentTag; // Store old tag for potential rollback
        try {
            // Optimistic UI update
            setCurrentTag(newTag);

            const response = await fetch(`http://localhost:5000/user/${userId}/playlist/update_tag`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ google_book_id: bookId, tag: newTag })
            });
            if (!response.ok) {
                // Rollback optimistic update on failure
                setCurrentTag(oldTag);
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.message || 'Failed to update tag');
            }
            const result = await response.json();
            // Optionally update state from result if backend modifies tag (e.g., validation)
            // setCurrentTag(result.tag); // Usually not needed if optimistic update is correct
            console.log("Tag updated successfully", result);
        } catch (error) {
            console.error('Error updating tag:', error);
            setCurrentTag(oldTag); // Rollback on error
            alert('Failed to update tag. Please try again.');
        }
    };

     const togglePlaylist = async () => {
        const userId = getUserId();
        if (!userId || !book?.google_book_id) return;
        const bookId = book.google_book_id;
        const url = `http://localhost:5000/user/${userId}/playlist`;
        const method = isInPlaylist ? 'DELETE' : 'POST';
        const finalUrl = isInPlaylist ? `${url}/${bookId}` : url;
        const originalIsInPlaylist = isInPlaylist; // For rollback
        const originalTag = currentTag;         // For rollback if adding

        try {
            // Optimistic UI update
             setIsInPlaylist(!isInPlaylist);
             if (!isInPlaylist) { // If adding, set tag optimistically
                setCurrentTag('save_later');
             }

             const options = {
                 method: method,
                 headers: method === 'POST' ? { 'Content-Type': 'application/json' } : {},
                 // Send full book details on POST
                 body: method === 'POST' ? JSON.stringify({ ...book, tag: 'save_later' }) : null,
            };
            const response = await fetch(finalUrl, options);
            if (!response.ok) {
                // Rollback optimistic update
                 setIsInPlaylist(originalIsInPlaylist);
                 if (!originalIsInPlaylist) setCurrentTag(originalTag);
                 const errData = await response.json().catch(() => ({}));
                 throw new Error(errData.message || `Failed to ${isInPlaylist ? 'remove from' : 'add to'} playlist`);
            }
            console.log(`Successfully ${isInPlaylist ? 'removed from' : 'added to'} playlist`);
             // No need to set state again if optimistic update worked
        } catch (error) {
            console.error('Error toggling playlist:', error);
            // Rollback optimistic update
             setIsInPlaylist(originalIsInPlaylist);
             if (!originalIsInPlaylist) setCurrentTag(originalTag);
             alert(`Failed to ${originalIsInPlaylist ? 'remove from' : 'add to'} playlist. Please try again.`);
        }
     };

    // --- Render Logic ---

    if (isLoading) {
        return (
            <div className="book-page loading-container">
                <div className="loading-indicator">Loading...</div>
            </div>
        );
    }

    if (error) { // Display error message
        return (
             <div className="book-page error-container">
                <h2 className="error-message">Error: {error}</h2>
                <button className="home-button" onClick={() => navigate('/')}>
                     Go Home
                 </button>
            </div>
        );
    }

    if (!book) { // Handle case where book is null after loading (e.g., 404)
        return (
            <div className="book-page not-found-container">
                <h2 className="not-found-message">Book not found!</h2>
                <button className="home-button" onClick={() => navigate('/')}>
                    Go Home
                </button>
            </div>
        );
    }

    // --- Extract Buy Link and Price from the book state ---
    // Adjust field names ('buyLink', 'listPrice') if your backend returns different names
    const buyLink = book.buyLink;
    const listPrice = book.listPrice; // Expecting object like { amount: 10.99, currencyCode: 'USD' }

    const displayPrice = listPrice && typeof listPrice.amount === 'number'
        ? `${listPrice.currencyCode || ''} ${listPrice.amount.toFixed(2)}` // Format price
        : null;

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
                            src={book.image_link || 'https://via.placeholder.com/300x450.png?text=No+Image'}
                            alt={`${book.title || 'Book'} cover`}
                            className="book-image"
                        />

                        {/* Like/Add to Playlist Actions */}
                        <div className="actions-container">
                            <button onClick={toggleLike} className={`action-button ${isLiked ? 'liked' : ''}`} >
                                <Heart fill={isLiked ? 'currentColor' : 'none'} size={18} /> {isLiked ? 'Unlike' : 'Like'}
                            </button>
                            {!isInPlaylist && (
                                <button onClick={togglePlaylist} className="action-button add-playlist" >
                                    <ListPlus size={18} /> Add to Playlist
                                </button>
                            )}
                        </div>

                         {/* Playlist Status/Actions Box (only if in playlist) */}
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

                        {/* Rating */}
                         <div className="rating-container">
                             <Star fill={book.rating ? "#FFD700" : "none"} stroke={book.rating ? "#FFD700" : "#ccc"}/> {/* Fill star if rating exists */}
                             <span style={{ marginLeft: '5px' }}>
                                 {typeof book.rating === 'number' ? `${book.rating.toFixed(1)} / 5` : 'No rating'}
                             </span>
                         </div>

                         {/* Genre */}
                        {book.genre && book.genre !== 'N/A' && (
                             <p className="genre-text">Genre: {book.genre}</p>
                        )}


                        {/* --- << BUY NOW & PRICE SECTION >> --- */}
                        <div className="buy-section"> {/* Add a CSS class for styling */}
                            {displayPrice && (
                                <span className="price-display"> {/* Add a CSS class */}
                                    <PriceTag size={18} /> Price: {displayPrice}
                                </span>
                            )}
                            {buyLink ? (
                                <a
                                    href={buyLink}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="buy-button" // Add a CSS class
                                >
                                    <ShoppingCart size={18} /> Buy Now
                                </a>
                            ) : (
                                // Optional: Show message if no buy link
                                <span className="buy-not-available">Purchase link unavailable</span>
                            )}
                        </div>
                        {/* --- << END BUY NOW & PRICE SECTION >> --- */}


                        {/* Synopsis */}
                        {book.synopsis && book.synopsis !== 'N/A' && (
                             <div>
                                 <h2 className="section-title">Synopsis</h2>
                                 <p className="synopsis-text">
                                     {book.synopsis}
                                 </p>
                             </div>
                        )}

                        {/* Similar Books */}
                        {similarBooks.length > 0 && (
                            <div>
                                <h2 className="section-title">Similar Books</h2>
                                <div className="similar-books-grid">
                                    {similarBooks.map((similarBook) => (
                                        <div
                                            key={similarBook.google_book_id}
                                            className="similar-book-card"
                                            onClick={() => navigate(`/book/${similarBook.google_book_id}`)}
                                            title={similarBook.title}
                                        >
                                            <img
                                                src={similarBook.image_link || 'https://via.placeholder.com/150x220.png?text=No+Image'}
                                                alt={similarBook.title}
                                                className="similar-book-image"
                                            />
                                            <p className="similar-book-title">{similarBook.title}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div> {/* End column-details */}

                </div> {/* End book-main-layout */}
            </div> {/* End content-wrapper */}
        </div> // End book-page
    );
};

export default BookPage;