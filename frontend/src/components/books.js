import React, { useState, useEffect, useCallback } from 'react';
import { Trash2, Search, BookOpen, PlusCircle, CheckSquare, Square, Bookmark } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

// Define the tags allowed in this view's playlist section
const ALLOWED_PLAYLIST_TAGS = ['save_later', 'completed', 'reading', 'dropped'];

const Books = () => {
    const navigate = useNavigate();
    const [books, setBooks] = useState([]); // Search results from left panel
    const [playlist, setPlaylist] = useState([]); // Will only hold items with ALLOWED_PLAYLIST_TAGS
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedTag, setSelectedTag] = useState('all'); // Filter within the allowed tags
    const [searchTrigger, setSearchTrigger] = useState(0); // Used to trigger search useEffect
    const [loadingBooks, setLoadingBooks] = useState(true); // Loading state for search results
    const [loadingPlaylist, setLoadingPlaylist] = useState(true); // Loading state for playlist

    // --- getUserId ---
    const getUserId = useCallback(() => {
        try {
            const userString = localStorage.getItem('user');
            if (!userString) return null;
            const user = JSON.parse(userString);
            return user ? user.id : null;
        } catch (error) {
            console.error("Error parsing user from localStorage:", error);
            return null;
        }
    }, []);

    const handleSearch = (e) => {
        e.preventDefault();
        setSearchTrigger(prev => prev + 1); // Trigger the search useEffect
    };

    // --- Fetch Search Results ---
    useEffect(() => {
        setLoadingBooks(true);
        const queryParam = searchQuery.trim()
            ? `?query=${encodeURIComponent(searchQuery)}`
            : '?query=subject:fiction'; // Default search if query is empty

        fetch(`http://localhost:5000/search${queryParam}`)
            .then((res) => {
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                return res.json();
            })
            .then((data) => setBooks(Array.isArray(data) ? data : []))
            .catch((error) => {
                 console.error("Error fetching books:", error);
                 setBooks([]); // Set empty on error
            })
            .finally(() => setLoadingBooks(false));
    }, [searchTrigger]); // Re-run when searchTrigger changes

    // --- Fetch Playlist (Filters by allowed tags) ---
    useEffect(() => {
        setLoadingPlaylist(true);
        const userId = getUserId();
        let isMounted = true;

        if (userId) {
            fetch(`http://localhost:5000/user/${userId}/playlist`)
                .then((res) => {
                    if (!res.ok) {
                        if (res.status === 404) return [];
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    return res.json();
                })
                .then((data) => {
                    if (isMounted) {
                        const validBooks = Array.isArray(data)
                            ? data.filter(book =>
                                ALLOWED_PLAYLIST_TAGS.includes(book.tag || 'save_later')
                              )
                            : [];
                        setPlaylist(validBooks);
                    }
                })
                .catch((error) => {
                    console.error("Error fetching playlist:", error);
                    if (isMounted) setPlaylist([]);
                })
                .finally(() => {
                    if (isMounted) setLoadingPlaylist(false);
                });
        } else {
            setPlaylist([]);
            setLoadingPlaylist(false);
        }
        return () => { isMounted = false; }; // Cleanup on unmount
    }, [getUserId]); // Re-run if getUserId changes (it won't due to useCallback)


    // --- Action Handlers ---
    const getBookId = (book) => book?.google_book_id || book?.id;

    const addToPlaylist = useCallback(async (book, tag = 'save_later') => {
        const bookId = getBookId(book);
        if (!bookId) { console.error("Cannot add book: Missing ID", book); alert("Cannot add this book: Missing ID."); return; }
        // Use a temporary check against current playlist state for immediate feedback
        if (playlist.some(item => getBookId(item) === bookId)) { console.log("Book already in playlist (frontend check)."); return; }
        const userId = getUserId();
        if (!userId) { alert("Please log in to add books."); navigate('/login'); return; }

        const bookPayload = {
            google_book_id: bookId, title: book.title || 'N/A',
            authors: Array.isArray(book.authors) ? book.authors.join(', ') : (book.authors || 'N/A'),
            synopsis: book.synopsis || 'N/A',
            rating: (r => r === 'N/A' || r == null ? null : parseFloat(r))(book.rating),
            genre: book.genre || 'N/A', image_link: book.image_link || null, tag: tag
        };

        try {
            const response = await fetch(`http://localhost:5000/user/${userId}/playlist`, {
                method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(bookPayload)
            });
            const data = await response.json();
            if (!response.ok) {
                // Handle conflict (already exists) specifically if backend sends 409 or specific message
                if (response.status === 409 || data.message?.includes('already in playlist')) {
                    alert("This book is already in your playlist (verified by server).");
                    // Ensure local state reflects this, only if the tag is allowed
                    if (!playlist.some(item => getBookId(item) === bookId) && ALLOWED_PLAYLIST_TAGS.includes(tag)) {
                        const bookToAdd = data.book ? { ...data.book, tag: data.book.tag || tag } : { ...bookPayload, google_book_id: bookId, tag: tag };
                        setPlaylist(prev => [...prev, bookToAdd]);
                    }
                    return;
                }
                throw new Error(data.message || `HTTP error! Status: ${response.status}`);
            }
            if (data.status === 'success' && data.book) {
                console.log("Book added, updating playlist state with:", data.book);
                const bookToAdd = { ...data.book, tag: data.book.tag || tag };
                // Only add to state if the tag is allowed in this view
                if (ALLOWED_PLAYLIST_TAGS.includes(bookToAdd.tag)) {
                    setPlaylist(prev => [...prev, bookToAdd]);
                } else {
                    console.log(`Book added with tag '${bookToAdd.tag}', but not displaying in this view.`);
                }
            } else { // Handle unexpected success response
                console.warn("Unexpected success response format:", data);
                 const bookToAdd = { ...bookPayload, google_book_id: bookId, tag: tag };
                if (ALLOWED_PLAYLIST_TAGS.includes(bookToAdd.tag)) {
                    setPlaylist(prev => [...prev, bookToAdd]);
                    alert("Book added, but encountered unexpected response format.");
                }
            }
        } catch (err) {
            console.error('Error adding book to playlist:', err); alert(`Error adding book: ${err.message}`);
        }
    }, [getUserId, playlist, navigate]);


    const removeFromPlaylist = useCallback(async (bookToRemove) => {
        const userId = getUserId();
        if (!userId) { alert("Please log in."); return; }
        const bookIdToRemove = getBookId(bookToRemove);
        if (!bookIdToRemove) { console.error("Cannot remove book: Missing ID"); return; }
        const originalPlaylist = [...playlist];
        setPlaylist(prev => prev.filter(book => getBookId(book) !== bookIdToRemove)); // Optimistic update
        try {
            const response = await fetch(`http://localhost:5000/user/${userId}/playlist/${bookIdToRemove}`, { method: "DELETE" });
            if (!response.ok) {
                setPlaylist(originalPlaylist); // Rollback
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }
            console.log(`Book ${bookIdToRemove} removed.`);
        } catch (err) {
            console.error('Error removing book:', err); alert(`Error removing book: ${err.message}`);
            setPlaylist(originalPlaylist); // Rollback
        }
    }, [getUserId, playlist]);


    const updatePlaylistTag = useCallback(async (book, newTag) => {
        const userId = getUserId();
        if (!userId) { alert("Please log in."); return; }
        const bookIdToUpdate = getBookId(book);
        if (!bookIdToUpdate) { console.error("Cannot update tag: Missing ID"); return; }
        const originalPlaylist = [...playlist]; // Store state before change
        const oldTag = playlist.find(item => getBookId(item) === bookIdToUpdate)?.tag;

        // Optimistic Update: Update state immediately, filtering based on allowed tags
        setPlaylist(prevPlaylist => {
            const updatedList = prevPlaylist.map(b =>
                getBookId(b) === bookIdToUpdate ? { ...b, tag: newTag } : b
            );
            // Re-filter based on allowed tags after the update
            return updatedList.filter(b => ALLOWED_PLAYLIST_TAGS.includes(b.tag || 'save_later'));
        });

        try {
             const response = await fetch(`http://localhost:5000/user/${userId}/playlist/update_tag`, {
                 method: "PUT", headers: { "Content-Type": "application/json" },
                 body: JSON.stringify({ google_book_id: bookIdToUpdate, tag: newTag })
             });
             const data = await response.json();
             if (!response.ok) {
                 setPlaylist(originalPlaylist); // Rollback on failure
                 throw new Error(data.message || `HTTP error! status: ${response.status}`);
             }
             console.log(`Tag updated for ${bookIdToUpdate} to ${data.tag || newTag}`);
         } catch (err) {
             console.error('Error updating tag:', err); alert(`Error updating tag: ${err.message}`);
             setPlaylist(originalPlaylist); // Rollback on error
         }
    }, [getUserId, playlist]);

    const toggleTag = (book, tag) => {
        if (book.tag === tag) return;
        updatePlaylistTag(book, tag);
    };

    // --- Filtering Logic ---
    // Define filteredBooks HERE, before the return statement
    const filteredBooks = books.filter(book =>
        (book.title || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
        (Array.isArray(book.authors) ? book.authors.join(', ') : book.authors || '').toLowerCase().includes(searchQuery.toLowerCase())
    );

    // Filter the playlist based on the selected tag button ('all', 'reading', etc.)
    const filteredPlaylist = selectedTag === 'all'
        ? playlist // 'playlist' state already contains only allowed tags
        : playlist.filter(book => book.tag === selectedTag);

    // Calculate tag counts based on the current (pre-filtered by ALLOWED_TAGS) playlist state
    const tagCounts = playlist.reduce((acc, book) => {
        const tag = book.tag || 'save_later'; // Should have a valid tag here
        acc[tag] = (acc[tag] || 0) + 1;
        return acc;
    }, {});
    tagCounts.all = playlist.length; // 'all' count is the total number of allowed books


    // --- Render Logic ---
    return (
        // Outer container div
        <div className="book-playlist-container" style={{ display: 'flex', height: 'calc(100vh - 70px)', /* Adjust based on actual navbar height */ backgroundColor: '#1a1a2e', color: 'white' }}>

            {/* Left Sidebar: Search Section */}
            <div className="search-section" style={{ width: '30%', minWidth: '300px', padding: '20px', backgroundColor: '#16213e', borderRight: '1px solid #2a2a4a', display: 'flex', flexDirection: 'column' }}>
                {/* Search Form */}
                <form onSubmit={handleSearch} style={{ marginBottom: '20px', flexShrink: 0 }}>
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Search books by title or author..."
                            style={{ flex: 1, padding: '10px', borderRadius: '5px', border: '1px solid #444', backgroundColor: '#1a1a2e', color: 'white' }}
                        />
                        <button
                            type="submit"
                            style={{ padding: '10px', borderRadius: '5px', border: 'none', backgroundColor: '#0f3460', color: 'white', cursor: 'pointer' }}
                            title="Search"
                        >
                            <Search size={20} />
                        </button>
                    </div>
                </form>

                {/* Books List (Search Results) */}
                {/* Use overflowY for scrolling */}
                <div className="books-list" style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '15px', paddingRight:'10px' /* Prevent scrollbar overlap */ }}>
                    {loadingBooks ? (
                        <p className="text-center p-3">Loading Search Results...</p>
                    ) : filteredBooks.length > 0 ? ( // <-- USING filteredBooks here
                        filteredBooks.map((book) => { // <-- USING filteredBooks here
                            const bookId = getBookId(book);
                            // Check if book is in the *current* playlist state
                            const isInPlaylist = playlist.some(item => getBookId(item) === bookId);
                            return (
                                // Book Card in Search Results
                                <div
                                    key={bookId || `search-${book.title}-${Math.random()}`} // More stable key
                                    style={{ backgroundColor: '#1a1a2e', borderRadius: '10px', padding: '10px', display: 'flex', gap: '10px', alignItems: 'center', cursor: 'pointer', border: '1px solid #2a2a4a' }}
                                    onClick={() => bookId && navigate(`/book/${bookId}`)}
                                    title={`View details for ${book.title || 'this book'}`}
                                >
                                    <img
                                        src={book.image_link || 'https://via.placeholder.com/60x90.png?text=N/A'}
                                        alt={book.title || 'Book cover'}
                                        style={{ width: '60px', height: '90px', objectFit: 'cover', borderRadius: '5px', flexShrink: 0, backgroundColor: '#333' /* Placeholder background */}}
                                        onError={(e) => { e.target.onerror = null; e.target.src='https://via.placeholder.com/60x90.png?text=N/A'; }}
                                    />
                                    <div style={{ flex: 1, overflow: 'hidden' }}>
                                        <h3 style={{ fontSize: '0.9em', margin: '0 0 5px 0', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{book.title || 'No Title'}</h3>
                                        <p style={{ fontSize: '0.8em', color: '#aaa', margin: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{Array.isArray(book.authors) ? book.authors.join(', ') : book.authors || 'Unknown Author'}</p>
                                        <button
                                            onClick={(event) => { event.stopPropagation(); if (!isInPlaylist) addToPlaylist(book); }} // Default tag is 'save_later'
                                            style={{
                                                backgroundColor: isInPlaylist ? '#2a2a4a' : '#0f3460', // Indicate added status
                                                border: 'none', borderRadius: '5px', padding: '5px 10px', marginTop: '5px',
                                                color: 'white', cursor: isInPlaylist ? 'default' : 'pointer', // Change cursor
                                                display: 'inline-flex', alignItems: 'center', gap: '5px', fontSize: '0.8em'
                                            }}
                                            disabled={isInPlaylist} // Disable button if added
                                            title={isInPlaylist ? "Already in your list" : "Add to 'Plan to Read'"}
                                        >
                                            <PlusCircle size={14} />
                                            {isInPlaylist ? "Added" : "Add"}
                                        </button>
                                    </div>
                                </div>
                            );
                        })
                    ) : (
                         // Message when no books are found
                         <p className="text-center p-3 text-muted" style={{color: '#888'}}>
                           {searchQuery.trim() ? 'No books found matching your search.' : 'Search for books to add to your list.'}
                         </p>
                     )}
                </div>
            </div>

            {/* Right Side - Playlist Section */}
            <div className="playlist-section" style={{ flex: 1, backgroundColor: '#1a1a2e', padding: '30px', overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
                {/* Playlist Header */}
                <h2 style={{ fontSize: '2em', marginBottom: '30px', borderBottom: '2px solid #2a2a4a', paddingBottom: '15px', flexShrink: 0 }}>My Reading List</h2>

                {/* Tag Filters */}
                <div className="tag-filters" style={{ display: 'flex', gap: '10px', /* Reduced gap */ marginBottom: '30px', flexWrap: 'wrap', backgroundColor: '#16213e', padding: '15px', borderRadius: '10px', flexShrink: 0 }}>
                     {/* Render 'all' button first */}
                     <button
                        key='all'
                        onClick={() => setSelectedTag('all')}
                        style={{ backgroundColor: selectedTag === 'all' ? '#0f3460' : 'transparent', border: '1px solid #0f3460', borderRadius: '20px', padding: '8px 16px', color: 'white', cursor: 'pointer', transition: 'all 0.2s ease' }}
                     >
                         All ({tagCounts.all || 0})
                     </button>
                     {/* Render buttons only for tags present in the (filtered) playlist */}
                     {ALLOWED_PLAYLIST_TAGS
                        .filter(tag => tagCounts[tag] > 0) // Only show tags with counts > 0
                        .map(tag => (
                         <button
                            key={tag}
                            onClick={() => setSelectedTag(tag)}
                            style={{ backgroundColor: selectedTag === tag ? '#0f3460' : 'transparent', border: '1px solid #0f3460', borderRadius: '20px', padding: '8px 16px', color: 'white', cursor: 'pointer', transition: 'all 0.2s ease' }}
                         >
                             {tag.charAt(0).toUpperCase() + tag.slice(1).replace('_', ' ')} ({tagCounts[tag]}) {/* Display count */}
                         </button>
                     ))}
                 </div>

                 {/* Playlist Books Grid */}
                 {/* Use overflowY for scrolling within this section */}
                 <div className="playlist-books" style={{ flex: 1, /* Let it take remaining space */ overflowY: 'auto', /* Allow internal scrolling */ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px', paddingBottom: '20px' }}>
                    {loadingPlaylist ? (
                        <p className="text-center p-3">Loading Reading List...</p>
                    ) : filteredPlaylist.length > 0 ? ( // Use filteredPlaylist which respects selectedTag
                        filteredPlaylist.map((book) => {
                            const bookId = getBookId(book);
                            return (
                                // Playlist Book Card
                                <div
                                    key={bookId || `playlist-${book.title}-${Math.random()}`}
                                    style={{ backgroundColor: '#16213e', borderRadius: '15px', padding: '20px', display: 'flex', flexDirection: 'column', gap: '15px', transition: 'transform 0.2s ease', cursor: 'pointer', minHeight:'280px' /* Ensure consistent card height */}}
                                    onClick={() => bookId && navigate(`/book/${bookId}`)}
                                    onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-5px)'}
                                    onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0px)'}
                                    title={`View details for ${book.title || 'this book'}`}
                                >
                                    {/* Card Content (Image + Text) */}
                                    <div style={{ display: 'flex', gap: '15px', flexShrink: 0 }}>
                                        <img
                                            src={book.image_link || 'https://via.placeholder.com/100x150.png?text=N/A'}
                                            alt={book.title || 'Book cover'}
                                            style={{ width: '100px', height: '150px', objectFit: 'cover', borderRadius: '8px', flexShrink: 0, backgroundColor: '#333' }}
                                            onError={(e) => { e.target.onerror = null; e.target.src='https://via.placeholder.com/100x150.png?text=N/A'; }}
                                        />
                                        <div style={{ flex: 1, overflow: 'hidden' }}>
                                            <h3 style={{ fontSize: '1.2em', marginBottom: '8px', color: '#fff', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>{book.title || 'No Title'}</h3>
                                            <p style={{ fontSize: '0.9em', color: '#aaa', marginBottom: '15px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{book.authors || 'Unknown Author'}</p>
                                        </div>
                                    </div>
                                    {/* Action Buttons */}
                                    {/* Use marginTop: auto to push actions to bottom */}
                                    <div className="book-actions" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: 'auto' }}>
                                        {/* Style buttons to indicate current tag */}
                                        <button title="Set as Reading" onClick={(e) => { e.stopPropagation(); toggleTag(book, 'reading');}} style={{ backgroundColor: book.tag === 'reading' ? '#0f3460' : 'transparent', border: '1px solid #0f3460', borderRadius: '5px', padding: '5px 10px', color: 'white', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.8em' }}> <BookOpen size={14} /> Reading </button>
                                        <button title="Set as Completed" onClick={(e) => { e.stopPropagation(); toggleTag(book, 'completed');}} style={{ backgroundColor: book.tag === 'completed' ? '#0f3460' : 'transparent', border: '1px solid #0f3460', borderRadius: '5px', padding: '5px 10px', color: 'white', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.8em' }}> <CheckSquare size={14} /> Done </button>
                                        <button title="Set as Dropped" onClick={(e) => { e.stopPropagation(); toggleTag(book, 'dropped');}} style={{ backgroundColor: book.tag === 'dropped' ? '#0f3460' : 'transparent', border: '1px solid #0f3460', borderRadius: '5px', padding: '5px 10px', color: 'white', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.8em' }}> <Square size={14} /> Drop </button>
                                        <button title="Set as Plan to Read" onClick={(e) => { e.stopPropagation(); toggleTag(book, 'save_later');}} style={{ backgroundColor: book.tag === 'save_later' ? '#0f3460' : 'transparent', border: '1px solid #0f3460', borderRadius: '5px', padding: '5px 10px', color: 'white', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.8em' }}> <Bookmark size={14} /> Plan to Read </button>
                                        {/* Push remove button to the right */}
                                        <button title="Remove from List" onClick={(e) => { e.stopPropagation(); removeFromPlaylist(book);}} style={{ backgroundColor: 'transparent', border: '1px solid #e94560', borderRadius: '5px', padding: '5px 10px', color: '#e94560', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.8em', marginLeft: 'auto' }}> <Trash2 size={14} /> Remove </button>
                                    </div>
                                </div>
                            );
                        })
                    ) : (
                        // Message when playlist is empty or filtered list is empty
                        <p className="text-center p-3 text-muted" style={{color: '#888'}}>
                          {selectedTag === 'all' ? 'Your reading list is empty. Add books from the search results!' : 'No books found with the selected status.'}
                        </p>
                     )}
                 </div>
            </div>
        </div>
    );
};

export default Books;