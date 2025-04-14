import React, { useState, useEffect } from 'react';
import {
  Heart,
  Filter,
  ArrowDownWideNarrow,
  ArrowUp01,
  Trash2
} from 'lucide-react';

const Liked = () => {
  const [likedBooks, setLikedBooks] = useState([]);
  const [sortCriteria, setSortCriteria] = useState('title');
  const [sortOrder, setSortOrder] = useState('asc');
  // Removed the potentially redundant 'liked' state

  // Retrieve logged in user (assumes user info is stored in localStorage)
  const getUserId = () => {
    const user = JSON.parse(localStorage.getItem('user'));
    return user ? user.id : null;
  };

  // Fetch liked books on load
  useEffect(() => {
    const userId = getUserId();
    if (!userId) {
      console.log("User not logged in. Cannot fetch liked books.");
      // Optionally set likedBooks to empty array or show a message
      setLikedBooks([]);
      return;
    }

    fetch(`http://localhost:5000/user/${userId}/liked`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json"
        // Add Authorization header if needed:
        // "Authorization": `Bearer ${yourAuthToken}`
      }
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
       })
      .then((data) => {
        // Assuming backend sends an array of book objects
        if (Array.isArray(data)) {
           // Process data from PSQL / Backend
           const formattedBooks = data.map((book) => ({
             // Ensure you get a consistent ID, prefer google_book_id if available
             google_book_id: book.google_book_id || book.id,
             title: book.title || 'No Title', // Add fallbacks
             authors: book.authors || 'Unknown Author',
             synopsis: book.synopsis || '',
             rating: book.rating,
             image_link: book.image_url || book.image_link, // Check backend field name
             // Ensure date_liked comes from backend and is valid
             dateLiked: book.date_liked ? new Date(book.date_liked) : new Date()
           }));
           setLikedBooks(formattedBooks);
        } else {
           // Handle cases where data is not an array (e.g., error object)
           console.error("Received unexpected data format:", data);
           setLikedBooks([]); // Set to empty to avoid errors
        }
      })
      .catch((error) => console.error("Error fetching liked books:", error));
  }, []); // Empty dependency array ensures this runs once on mount

  // Remove book from liked books with backend integration (non-optimistic)
  const removeFromLikedBooks = async (bookToRemove) => {
    const userId = getUserId();
    if (!userId) {
      alert("Please log in to manage your liked books.");
      return;
    }

    // Ensure we have the correct ID to send to the backend
    const bookId = bookToRemove.google_book_id;
    if (!bookId) {
        console.error("Cannot remove book: Missing google_book_id", bookToRemove);
        alert("Error: Could not identify the book to remove.");
        return;
    }

    try {
      console.log(`Attempting to delete book with ID: ${bookId} for user: ${userId}`); // Debug log
      const response = await fetch(`http://localhost:5000/user/${userId}/liked/${bookId}`, {
        method: "DELETE",
        headers: {
            "Content-Type": "application/json"
            // Add Authorization header if needed
        }
      });

      const data = await response.json(); // Try to parse JSON regardless of status

      if (response.ok && data.status === 'success') {
        console.log("Backend confirmed deletion:", data.message); // Debug log
        // Update local state only AFTER backend confirms success
        setLikedBooks(prevLikedBooks =>
          prevLikedBooks.filter(book => book.google_book_id !== bookId)
        );
      } else {
        // Handle backend errors or non-success status
        console.error("Failed to delete book:", data.message || `HTTP status ${response.status}`); // Debug log
        alert(data.message || `Failed to remove book. Server responded with status ${response.status}.`);
      }
    } catch (err) {
      // Handle network errors or issues with the fetch/JSON parsing
      console.error("Error removing book from liked:", err);
      alert("An error occurred while trying to remove the book. Please check your connection and try again.");
    }
  };


  // Sort liked books function (remains the same)
  const sortLikedBooks = (criteria) => {
    if (sortCriteria === criteria) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortCriteria(criteria);
      setSortOrder('asc');
    }
  };

  // Process and sort liked books (remains the same, uses likedBooks state)
  // Use slice() to create a copy before sorting, preventing direct state mutation
  const processedLikedBooks = [...likedBooks].sort((a, b) => {
    let comparison = 0;
    switch(sortCriteria) {
      case 'title':
        // Handle potential null/undefined titles gracefully
        comparison = (a.title || '').localeCompare(b.title || '');
        break;
      case 'dateLiked':
        // Ensure dates are valid before comparing
        const dateA = a.dateLiked instanceof Date ? a.dateLiked.getTime() : 0;
        const dateB = b.dateLiked instanceof Date ? b.dateLiked.getTime() : 0;
        comparison = dateA - dateB;
        break;
      default:
        comparison = 0;
    }
    return sortOrder === 'asc' ? comparison : -comparison;
  });

  // ----- JSX Rendering -----
  return (
    <div style={{
      display: 'flex',
      height: '100vh',
      backgroundColor: '#1a1a2e',
      color: 'white'
    }}>
      {/* Sidebar - Sorting */}
      <div
        className="favorites-sidebar"
        style={{
          width: '250px',
          backgroundColor: '#16213e',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        <h3 style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
          <Filter style={{ marginRight: '10px' }} /> Sorting
        </h3>

        {/* Sorting Options */}
        <div>
          <h4>Sort By</h4>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {[
              { criteria: 'title', label: 'Title' },
              { criteria: 'dateLiked', label: 'Date Liked' }
            ].map(({ criteria, label }) => (
              <button
                key={criteria} // Key is stable here
                onClick={() => sortLikedBooks(criteria)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  background: sortCriteria === criteria ? '#0f3460' : 'transparent',
                  border: 'none',
                  color: 'white',
                  padding: '10px',
                  borderRadius: '5px',
                  marginBottom: '5px',
                  cursor: 'pointer', // Add cursor pointer
                  textAlign: 'left' // Ensure text alignment
                }}
              >
                {label}
                {sortCriteria === criteria && (
                  sortOrder === 'asc'
                    ? <ArrowDownWideNarrow style={{ marginLeft: 'auto' }} /> // Use margin auto for alignment
                    : <ArrowUp01 style={{ marginLeft: 'auto' }} />
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Liked Books Content */}
      <div
        className="favorites-content"
        style={{
          flex: 1,
          backgroundColor: '#0f3460',
          padding: '20px',
          overflowY: 'auto'
        }}
      >
        <h2 style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
          <Heart style={{ marginRight: '10px', fill: 'red', color: 'red' }} />
          Liked Books
        </h2>

        {processedLikedBooks.length === 0 ? (
          <div
            style={{
              textAlign: 'center',
              color: '#aaa',
              marginTop: '50px'
            }}
          >
           { getUserId() ? "No liked books yet. Start liking some!" : "Please log in to see your liked books."}
          </div>
        ) : (
          processedLikedBooks.map((book) => ( // Removed index from map params
            <div
              // *** USE UNIQUE BOOK ID FOR KEY ***
              key={book.google_book_id || book.title} // Fallback to title if ID is missing, though ID is preferred
              style={{
                display: 'flex',
                alignItems: 'center',
                backgroundColor: '#16213e',
                borderRadius: '10px',
                padding: '15px',
                marginBottom: '10px',
                transition: 'all 0.3s ease'
              }}
              className="liked-book-item"
            >
              <img
                src={book.image_link || 'https://via.placeholder.com/80x120'} // Provide fallback URL directly
                alt={book.title}
                style={{
                  width: '80px',
                  height: '120px',
                  objectFit: 'cover',
                  marginRight: '15px',
                  borderRadius: '5px'
                }}
                // Simplified onError: just set to placeholder
                onError={(e) => { e.target.src = 'https://via.placeholder.com/80x120'; }}
              />

              <div style={{ flex: 1 }}>
                <h3 style={{ margin: 0, fontSize: '18px' }}>{book.title}</h3>
                <p style={{ margin: '5px 0', color: '#aaa' }}>{book.authors}</p>

                {/* Ensure dateLiked is a valid Date object before calling methods */}
                {book.dateLiked instanceof Date && !isNaN(book.dateLiked) && (
                   <p style={{ margin: '5px 0', fontSize: '12px', color: '#777' }}>
                     Liked on: {book.dateLiked.toLocaleDateString()}
                   </p>
                )}
              </div>

              <div style={{ display: 'flex', alignItems: 'center' }}>
                <button
                  onClick={() => removeFromLikedBooks(book)} // Pass the whole book object
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'white',
                    cursor: 'pointer' // Add cursor pointer
                  }}
                  aria-label={`Remove ${book.title} from liked books`} // Accessibility
                >
                  <Trash2 color="red" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default Liked;