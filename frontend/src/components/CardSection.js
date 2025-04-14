import React, { useState, useEffect } from 'react';
import { Heart, ListPlus } from 'lucide-react';
import './CardSection.css';
import { useNavigate } from 'react-router-dom';

const CardSection = () => {
  const navigate = useNavigate();
  const [books, setBooks] = useState([]);
  // --- Refactored State: Use maps for efficient lookups ---
  const [likedBooks, setLikedBooks] = useState({}); // Map: { google_book_id: true }
  const [playlistBooks, setPlaylistBooks] = useState({}); // Map: { google_book_id: true }
  const [loading, setLoading] = useState(true); // Added loading state

  // --- Helper Functions ---
  const getUserId = () => {
    try {
      const userString = localStorage.getItem('user');
      const user = userString ? JSON.parse(userString) : null;
      return user ? user.id : null;
    } catch (error) {
      console.error("Error parsing user from localStorage:", error);
      return null;
    }
  };

  // Use consistent ID (prefer google_book_id if available)
  const getBookId = (book) => book.google_book_id || book.id;

  // --- Fetch Initial Data ---
  useEffect(() => {
    const userId = getUserId();
    setLoading(true);

    // Fetch books displayed in the section
    const fetchBooks = fetch("http://localhost:5000/data") // Assuming /data provides the books for this section
      .then((res) => {
           if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
           return res.json();
      })
       .then((data) => setBooks(Array.isArray(data) ? data : []))
       .catch((error) => {
            console.error("Error fetching books:", error);
            setBooks([]); // Set empty on error
       });


    // Fetch liked status if user exists
    const fetchLiked = userId
      ? fetch(`http://localhost:5000/user/${userId}/liked`)
          .then((res) => {
               if (!res.ok) { if (res.status === 404) return []; throw new Error(`HTTP error! status: ${res.status}`); } // Handle 404 as empty
               return res.json();
          })
          .then((data) => {
            const newLikedMap = {};
             if(Array.isArray(data)){
                 data.forEach((book) => {
                    const bookId = getBookId(book);
                    if(bookId) newLikedMap[bookId] = true;
                });
             }
            setLikedBooks(newLikedMap);
          })
          .catch((error) => console.error("Error fetching liked books:", error))
      : Promise.resolve(); // Resolve immediately if no user


    // Fetch playlist status if user exists
    const fetchPlaylist = userId
      ? fetch(`http://localhost:5000/user/${userId}/playlist`)
          .then((res) => {
              if (!res.ok) { if (res.status === 404) return []; throw new Error(`HTTP error! status: ${res.status}`); } // Handle 404 as empty
              return res.json();
          })
          .then((data) => {
            const newPlaylistMap = {};
            if(Array.isArray(data)){
                data.forEach((book) => {
                   const bookId = getBookId(book);
                   if(bookId) newPlaylistMap[bookId] = true;
                });
            }
            setPlaylistBooks(newPlaylistMap);
          })
          .catch((error) => console.error("Error fetching playlist:", error))
      : Promise.resolve(); // Resolve immediately if no user

    // Wait for all fetches to complete
    Promise.all([fetchBooks, fetchLiked, fetchPlaylist])
        .catch(err => console.error("Error during initial data fetch:", err)) // Catch errors from Promise.all itself if needed
        .finally(() => setLoading(false));

  }, []); // Run only on mount

  // --- Action Handlers (Add/Remove) ---

  const handleAdd = async (listType, book) => {
      const userId = getUserId();
      if (!userId) {
          alert(`Please log in to add books to ${listType}.`);
          return;
      }
      const bookId = getBookId(book);
       if (!bookId) {
            console.error("Cannot add book without ID:", book);
            alert("Error: Book ID is missing.");
            return;
      }

      try {
        // Ensure all needed book properties are included
        const bookPayload = {
            google_book_id: bookId,
            title: book.title || 'N/A',
             // Ensure authors is a string for backend if needed
            authors: Array.isArray(book.authors) ? book.authors.join(', ') : book.authors || 'N/A',
            synopsis: book.synopsis || 'N/A',
            rating: book.rating || null,
            genre: book.genre || 'N/A',
            image_link: book.image_link || null,
            // Add default tag only for playlist
            ...(listType === 'playlist' && { tag: 'save_later' })
        };

        const response = await fetch(`http://localhost:5000/user/${userId}/${listType}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bookPayload)
        });

        if (!response.ok) {
             // Handle specific errors like 409 Conflict (already exists)
             if(response.status === 409) {
                 const errorData = await response.json().catch(()=>({}));
                 console.warn(`Book ${bookId} already in ${listType} (backend check).`);
                 // Ensure local state matches backend if conflict occurs
                 if (listType === 'playlist') setPlaylistBooks(prev => ({ ...prev, [bookId]: true }));
                 if (listType === 'liked') setLikedBooks(prev => ({ ...prev, [bookId]: true }));
                 return; // Don't show generic error
             }
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

         // Update local state map directly after successful API call
         if (listType === 'playlist') {
             setPlaylistBooks(prev => ({ ...prev, [bookId]: true }));
         } else if (listType === 'liked') {
             setLikedBooks(prev => ({ ...prev, [bookId]: true }));
         }

      } catch (err) {
        console.error(`Error adding book to ${listType}:`, err);
        alert(`Error adding book to ${listType}: ${err.message}`);
      }
  };

  const handleRemove = async (listType, book) => {
       const userId = getUserId();
       if (!userId) {
           alert(`Please log in to remove books from ${listType}.`);
           return;
       }
       const bookId = getBookId(book);
        if (!bookId) {
            console.error("Cannot remove book without ID:", book);
            alert("Error: Book ID is missing.");
            return;
       }

       try {
           const response = await fetch(`http://localhost:5000/user/${userId}/${listType}/${bookId}`, {
               method: "DELETE",
               headers: { "Content-Type": "application/json" }
           });

           if (!response.ok && response.status !== 404) { // Ignore 404 (already removed)
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
           }

           // Update local state map directly after successful API call (or 404)
            if (listType === 'playlist') {
                setPlaylistBooks(prev => {
                    const newMap = {...prev};
                    delete newMap[bookId];
                    return newMap;
                });
            } else if (listType === 'liked') {
                 setLikedBooks(prev => {
                    const newMap = {...prev};
                    delete newMap[bookId];
                    return newMap;
                });
            }

       } catch (err) {
           console.error(`Error removing book from ${listType}:`, err);
           alert(`Error removing book from ${listType}: ${err.message}`);
       }
  };


  // --- Render Logic ---

  if (loading) {
      return <div>Loading books...</div>; // Or a spinner component
  }

  return (
    <div>
      <h2 className="font-weight-bold text-white">Top rated books</h2> {/* Make sure styles are defined */}
      <div className="card-container text-white"> {/* Make sure styles are defined */}
        {books.map((book) => {
            const bookId = getBookId(book); // Get consistent ID
            const isLiked = !!likedBooks[bookId]; // Check map for liked status
            const isInPlaylist = !!playlistBooks[bookId]; // Check map for playlist status

            if (!bookId) {
                console.warn("Skipping book render due to missing ID:", book);
                return null; // Don't render card if book ID is missing
            }

            return (
              <div
                className="card text-white" // Make sure styles are defined
                key={bookId} // Use consistent ID
                // *** CARD onClick for navigation ***
                onClick={() => navigate(`/book/${bookId}`)}
                style={{ cursor: 'pointer', backgroundColor: '#333', margin: '10px', padding: '10px', borderRadius: '8px' }} // Example inline styles
              >
                <img
                  src={book.image_link || 'https://via.placeholder.com/200x300.png?text=No+Image'}
                  alt={book.title || 'Book cover'}
                  style={{ width: '100%', height: '250px', objectFit: 'cover', borderRadius: '4px' }} // Example image styles
                  onError={(e) => { e.target.onerror = null; e.target.src = 'https://via.placeholder.com/200x300.png?text=No+Image'; }}
                />
                <div className='d-flex justify-content-end align-items-center' style={{ gap: "10px", marginTop: "10px" }}>
                  {/* --- Like Button --- */}
                  <button
                    // *** BUTTON onClick with stopPropagation ***
                    onClick={(event) => {
                      event.stopPropagation(); // <-- FIX: Stop bubbling
                      if (isLiked) {
                        handleRemove('liked', book);
                      } else {
                        handleAdd('liked', book);
                      }
                    }}
                    className='text-white bg-transparent border-0 p-0' // Remove padding if needed
                    title={isLiked ? "Unlike" : "Like"}
                    style={{ lineHeight: '1' }} // Adjust line height if needed
                  >
                    <Heart
                      color={isLiked ? 'red' : 'white'}
                      fill={isLiked ? 'red' : 'none'}
                      strokeWidth={2}
                      size={24} // Example size
                    />
                  </button>
                   {/* --- Playlist Button --- */}
                  <button
                     // *** BUTTON onClick with stopPropagation ***
                    onClick={(event) => {
                      event.stopPropagation(); // <-- FIX: Stop bubbling
                       if (isInPlaylist) {
                        handleRemove('playlist', book);
                      } else {
                        handleAdd('playlist', book);
                      }
                    }}
                    className='text-white bg-transparent border-0 p-0' // Remove padding if needed
                    title={isInPlaylist ? "Remove from Playlist" : "Add to Playlist"}
                     style={{ lineHeight: '1' }} // Adjust line height if needed
                  >
                    <ListPlus
                      // onClick should be on the BUTTON, not here
                      color={isInPlaylist ? 'lime' : 'white'} // Changed color for visibility
                      // fill might look odd, usually just color change is enough
                      strokeWidth={2}
                       size={24} // Example size
                    />
                  </button>
                </div>
                <h3 className='text-white mt-2' style={{ fontSize: '1.1em' }}>{book.title || 'No Title'}</h3>
                <p className='text-white' style={{ fontSize: '0.9em' }}>
                   {/* Ensure authors is displayed correctly */}
                  <strong>Authors:</strong> {Array.isArray(book.authors) ? book.authors.join(', ') : book.authors || 'N/A'}<br />
                  <strong>Rating:</strong> {book.rating || 'N/A'}
                </p>
              </div>
            );
        })}
      </div>
    </div>
  );
};

export default CardSection;