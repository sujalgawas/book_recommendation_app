import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Heart, ListPlus, ArrowLeft, Star, BookOpen, CheckSquare, Square, Bookmark, Trash2 } from 'lucide-react';

// Import the CSS file
import './BookPage.css';

const BookPage = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [book, setBook] = useState(null);
    const [similarBooks, setSimilarBooks] = useState([]);
    const [isLiked, setIsLiked] = useState(false);
    const [isInPlaylist, setIsInPlaylist] = useState(false);
    const [currentTag, setCurrentTag] = useState('save_later');
    const [isLoading, setIsLoading] = useState(true);

    // --- getUserId and useEffect logic remains the same ---
    const getUserId = () => {
        const user = JSON.parse(localStorage.getItem('user') || '{}');
        return user.id;
    };

    useEffect(() => {
        const fetchData = async () => {
            setIsLoading(true);
            try {
                // 🆕 Use the new backend route that fetches by Google Book ID
                const bookResponse = await fetch(`http://localhost:5000/book/${id}`);
                if (!bookResponse.ok) throw new Error('Failed to fetch book details');
                
                const bookData = await bookResponse.json();
                setBook(bookData);
        
                // Fetch similar books based on genre
                const similarResponse = await fetch(`http://localhost:5000/search?query=${encodeURIComponent(bookData.genre || 'fiction')}&num_results=5`);
                const similarData = await similarResponse.json();
                if (similarData) {
                    setSimilarBooks(similarData.filter((b) => b.google_book_id !== id).slice(0, 4));
                }
        
                // Check user-related data
                const userId = getUserId();
                if (userId) {
                    const likedResponse = await fetch(`http://localhost:5000/user/${userId}/liked`);
                    if (likedResponse.ok) {
                        const likedData = await likedResponse.json();
                        setIsLiked(likedData.some((b) => b.google_book_id === id));
                    }
        
                    const playlistResponse = await fetch(`http://localhost:5000/user/${userId}/playlist`);
                    if (playlistResponse.ok) {
                        const playlistData = await playlistResponse.json();
                        const playlistBook = playlistData.find((b) => b.google_book_id === id);
                        if (playlistBook) {
                            setIsInPlaylist(true);
                            setCurrentTag(playlistBook.tag || 'save_later');
                        } else {
                            setIsInPlaylist(false);
                        }
                    }
                }
            } catch (error) {
                console.error('Error fetching data:', error);
            } finally {
                setIsLoading(false);
            }
        };        
        fetchData();
    }, [id, navigate]);

    // --- Action Handlers remain the same ---
     const toggleLike = async () => {
        const userId = getUserId();
        if (!userId || !book) return;
        const url = `http://localhost:5000/user/${userId}/liked`;
        const method = isLiked ? 'DELETE' : 'POST';
        const finalUrl = isLiked ? `${url}/${book.google_book_id}` : url;
        try {
            const options = {
                 method: method,
                 headers: method === 'POST' ? { 'Content-Type': 'application/json' } : {},
                 body: method === 'POST' ? JSON.stringify(book) : null,
             };
            const response = await fetch(finalUrl, options);
             if (!response.ok) throw new Error(`Failed to ${isLiked ? 'unlike' : 'like'} book`);
            setIsLiked(!isLiked);
        } catch (error) {
            console.error('Error toggling like:', error);
            alert(`Failed to ${isLiked ? 'unlike' : 'like'} book. Please try again.`);
        }
    };

    const updatePlaylistTag = async (newTag) => {
        const userId = getUserId();
         if (!userId || !book) return;
        try {
            const response = await fetch(`http://localhost:5000/user/${userId}/playlist/update_tag`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ google_book_id: book.google_book_id, tag: newTag })
            });
             if (!response.ok) throw new Error('Failed to update tag');
             const result = await response.json();
            setCurrentTag(result.tag);
        } catch (error) {
            console.error('Error updating tag:', error);
            alert('Failed to update tag. Please try again.');
        }
    };

     const togglePlaylist = async () => {
         const userId = getUserId();
         if (!userId || !book) return;
         const url = `http://localhost:5000/user/${userId}/playlist`;
         const method = isInPlaylist ? 'DELETE' : 'POST';
         const finalUrl = isInPlaylist ? `${url}/${book.google_book_id}` : url;
         try {
              const options = {
                 method: method,
                 headers: method === 'POST' ? { 'Content-Type': 'application/json' } : {},
                 body: method === 'POST' ? JSON.stringify({ ...book, tag: 'save_later' }) : null,
             };
             const response = await fetch(finalUrl, options);
             if (!response.ok) throw new Error(`Failed to ${isInPlaylist ? 'remove from' : 'add to'} playlist`);
             if (isInPlaylist) { setIsInPlaylist(false); }
             else { setIsInPlaylist(true); setCurrentTag('save_later'); }
         } catch (error) {
             console.error('Error toggling playlist:', error);
             alert(`Failed to ${isInPlaylist ? 'remove from' : 'add to'} playlist. Please try again.`);
         }
     };

    // --- Render Logic ---

    if (isLoading) {
        return (
            <div className="book-page loading-container"> {/* Use CSS classes */}
                 <div className="loading-indicator">Loading...</div>
            </div>
        );
    }
    
     if (!book) {
         return (
             <div className="book-page not-found-container"> {/* Use CSS classes */}
                 <h2 className="not-found-message">Book not found!</h2>
                 <button className="home-button" onClick={() => navigate('/')}>
                     Go Home
                 </button>
             </div>
         );
     }


    // Main Page Content
    return (
        <div className="book-page"> {/* Use CSS class */}
            <div className="content-wrapper"> {/* Use CSS class */}

                {/* Back Button */}
                <button className="back-button" onClick={() => navigate(-1)} > {/* Use CSS class */}
                    <ArrowLeft size={18} />
                    Back
                </button>

                <div className="book-main-layout"> {/* Use CSS class */}

                    {/* --- Column 1: Book Cover and Actions --- */}
                    <div className="column-image-actions"> {/* Use CSS class */}
                        <img
                            src={book.image_link || 'https://via.placeholder.com/300x450.png?text=No+Image'}
                            alt={book.title}
                            className="book-image" // Use CSS class
                        />

                        {/* --- Action Buttons --- */}
                        <div className="actions-container"> {/* Use CSS class */}
                            <button
                                onClick={toggleLike}
                                // Conditionally add 'liked' class
                                className={`action-button ${isLiked ? 'liked' : ''}`}
                            >
                                <Heart fill={isLiked ? 'currentColor' : 'none'} size={18} />
                                {isLiked ? 'Unlike' : 'Like'}
                            </button>

                            {!isInPlaylist && (
                                <button
                                    onClick={togglePlaylist}
                                    className="action-button add-playlist" // Use CSS class
                                >
                                    <ListPlus size={18} />
                                    Add to Playlist
                                </button>
                            )}
                        </div>

                        {/* --- Playlist Tag Buttons (only show if in playlist) --- */}
                         {isInPlaylist && (
                            <div className="playlist-status-box"> {/* Use CSS class */}
                                <h5>Playlist Status:</h5>
                                 <div className="playlist-buttons-container"> {/* Use CSS class */}
                                    {/* Add tag-specific class and active class conditionally */}
                                     <button onClick={() => updatePlaylistTag('reading')} className={`playlist-tag-button tag-reading ${currentTag === 'reading' ? 'active' : ''}`}>
                                         <BookOpen size={16} /> Reading
                                     </button>
                                     <button onClick={() => updatePlaylistTag('completed')} className={`playlist-tag-button tag-completed ${currentTag === 'completed' ? 'active' : ''}`}>
                                         <CheckSquare size={16} /> Completed
                                     </button>
                                     <button onClick={() => updatePlaylistTag('dropped')} className={`playlist-tag-button tag-dropped ${currentTag === 'dropped' ? 'active' : ''}`}>
                                         <Square size={16} /> Dropped
                                     </button>
                                      <button onClick={() => updatePlaylistTag('save_later')} className={`playlist-tag-button tag-save_later ${currentTag === 'save_later' ? 'active' : ''}`}>
                                         <Bookmark size={16} /> Save Later
                                     </button>
                                      <button onClick={togglePlaylist} title="Remove from Playlist" className="playlist-tag-button remove-button"> {/* Use CSS classes */}
                                         <Trash2 size={16} /> Remove
                                     </button>
                                 </div>
                             </div>
                         )}
                    </div>

                    {/* --- Column 2: Book Details --- */}
                    <div className="column-details"> {/* Use CSS class */}
                        <h1 className="book-title">{book.title || 'Title Not Available'}</h1> {/* Use CSS class */}
                        <p className="book-authors">by {book.authors || 'Unknown Author'}</p> {/* Use CSS class */}

                         <div className="rating-container"> {/* Use CSS class */}
                             <Star/>
                             <span>
                                 {book.rating === 'N/A' || book.rating == null ? 'No ratings yet' : `${book.rating} / 5`}
                             </span>
                         </div>

                         <div>
                             <h2 className="section-title">Genre</h2> {/* Use CSS class */}
                             <span className="genre-text">{book.genre || 'N/A'}</span> {/* Use CSS class */}
                         </div>

                        <div>
                            <h2 className="section-title">Synopsis</h2> {/* Use CSS class */}
                            <p className="synopsis-text"> {/* Use CSS class */}
                                {book.synopsis || 'No synopsis available.'}
                            </p>
                        </div>
                        {/* --- Similar Books Section --- */}
                         {similarBooks.length > 0 && (
                            <div>
                                <h2 className="section-title">Similar Books</h2> {/* Use CSS class */}
                                <div className="similar-books-grid"> {/* Use CSS class */}
                                    {similarBooks.map((similarBook) => (
                                        <div
                                            key={similarBook.google_book_id}
                                            className="similar-book-card" // Use CSS class
                                            onClick={() => navigate(`/book/${similarBook.google_book_id}`)}
                                            title={similarBook.title}
                                        >
                                            <img
                                                src={similarBook.image_link || 'https://via.placeholder.com/150x220.png?text=No+Image'}
                                                alt={similarBook.title}
                                                className="similar-book-image" // Use CSS class
                                            />
                                            <p className="similar-book-title">{similarBook.title}</p> {/* Use CSS class */}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div> {/* End mainLayout */}
            </div> {/* End contentWrapper */}
        </div> // End page
    );
};

export default BookPage;