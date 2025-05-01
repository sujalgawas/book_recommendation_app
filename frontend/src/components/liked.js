import React, { useState, useEffect, useCallback } from 'react';
import {
  Heart,
  Filter,
  ArrowDownWideNarrow,
  ArrowUp01,
  Trash2,
  X // Import X icon for closing the mobile sidebar
} from 'lucide-react';

// --- Simple Custom Hook for Media Query ---
function useMediaQuery(query) {
  // Use useCallback to memoize the media query list creation and listener logic
  const getMatches = useCallback((q) => window.matchMedia(q).matches, []);
  const [matches, setMatches] = useState(getMatches(query));

  useEffect(() => {
    const mediaQueryList = window.matchMedia(query);
    const listener = (event) => setMatches(event.matches);

    // Initial check in case the state is already correct
    setMatches(getMatches(query));

    // Add listener using the recommended addEventListener method
    try {
      mediaQueryList.addEventListener('change', listener);
    } catch (e) {
      // Fallback for older browsers
      mediaQueryList.addListener(listener);
    }


    // Cleanup function
    return () => {
      try {
         mediaQueryList.removeEventListener('change', listener);
      } catch(e) {
         // Fallback for older browsers
         mediaQueryList.removeListener(listener);
      }
    };
  }, [query, getMatches]); // Add getMatches to dependency array

  return matches;
}
// --- End Custom Hook ---


const Liked = () => {
  const [likedBooks, setLikedBooks] = useState([]);
  const [sortCriteria, setSortCriteria] = useState('title');
  const [sortOrder, setSortOrder] = useState('asc');
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false); // State for mobile sidebar visibility

  // Use the hook to detect mobile screen size (adjust breakpoint as needed)
  const isMobile = useMediaQuery('(max-width: 768px)');

  // --- Functions (getUserId, fetch, remove, sort - mostly unchanged) ---

  const getUserId = useCallback(() => { // useCallback for stable reference
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

  // Fetch liked books on load
  useEffect(() => {
    const userId = getUserId();
    if (!userId) {
      console.log("User not logged in. Cannot fetch liked books.");
      setLikedBooks([]);
      return;
    }

    let isMounted = true; // Flag to prevent state update on unmounted component

    fetch(`http://localhost:5000/user/${userId}/liked`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json"
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
        if (!isMounted) return; // Don't update state if component unmounted

        if (Array.isArray(data)) {
          const formattedBooks = data.map((book) => ({
            google_book_id: book.google_book_id || book.id || `fallback-${Math.random()}`, // Add stronger fallback
            title: book.title || 'No Title',
            authors: book.authors || 'Unknown Author',
            synopsis: book.synopsis || '',
            rating: book.rating,
            image_link: book.image_url || book.image_link,
            dateLiked: book.date_liked ? new Date(book.date_liked) : new Date()
          })).filter(book => book.google_book_id); // Ensure books have an ID for key prop

          setLikedBooks(formattedBooks);
        } else {
          console.error("Received unexpected data format:", data);
          setLikedBooks([]);
        }
      })
      .catch((error) => {
          if (isMounted) {
            console.error("Error fetching liked books:", error);
            // Optionally set an error state here to show the user
          }
      });

      return () => {
          isMounted = false; // Set flag when component unmounts
      };
  }, [getUserId]); // Add getUserId as dependency

  // Remove book from liked books
  const removeFromLikedBooks = async (bookToRemove) => {
    const userId = getUserId();
    if (!userId) {
      alert("Please log in to manage your liked books.");
      return;
    }
    const bookId = bookToRemove.google_book_id;
    if (!bookId) {
      console.error("Cannot remove book: Missing google_book_id", bookToRemove);
      alert("Error: Could not identify the book to remove.");
      return;
    }

    // --- (Keep the fetch DELETE logic as it was) ---
    try {
        console.log(`Attempting to delete book with ID: ${bookId} for user: ${userId}`); // Debug log
        const response = await fetch(`http://localhost:5000/user/${userId}/liked/${bookId}`, {
        method: "DELETE",
        headers: {
            "Content-Type": "application/json"
            // Add Authorization header if needed
        }
        });

        // Try to parse JSON, but handle potential empty responses on success (204 No Content)
        let data = {};
        if (response.status !== 204) {
            try {
                data = await response.json();
            } catch (jsonError) {
                // If JSON parsing fails but status is ok, treat as success (maybe empty response)
                if (!response.ok) {
                     console.error("Failed to parse JSON response:", jsonError);
                     throw new Error(`Server responded with status ${response.status} but failed to parse JSON.`);
                }
                console.warn("Response okay, but no JSON body or parse error. Assuming success.");
                data = { status: 'success', message: 'Book removed (no content)' }; // Simulate success
            }
        } else {
             // Handle 204 No Content as success
             data = { status: 'success', message: 'Book removed (204 No Content)' };
        }


        if (response.ok || response.status === 204) {
        console.log("Backend confirmed deletion:", data.message || `Status ${response.status}`); // Debug log
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

  // Sort liked books function
  const sortLikedBooks = (criteria) => {
    if (sortCriteria === criteria) {
      setSortOrder(prevOrder => (prevOrder === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortCriteria(criteria);
      setSortOrder('asc');
    }
    // Optional: Close mobile sidebar after sorting
    if (isMobile) {
        // setIsMobileSidebarOpen(false); // Uncomment if you want this behaviour
    }
  };

  // Process and sort liked books
  const processedLikedBooks = [...likedBooks].sort((a, b) => {
    let comparison = 0;
    switch (sortCriteria) {
      case 'title':
        comparison = (a.title || '').localeCompare(b.title || '');
        break;
      case 'dateLiked':
        const dateA = a.dateLiked instanceof Date && !isNaN(a.dateLiked) ? a.dateLiked.getTime() : 0;
        const dateB = b.dateLiked instanceof Date && !isNaN(b.dateLiked) ? b.dateLiked.getTime() : 0;
        comparison = dateA - dateB;
        break;
      default:
        comparison = 0;
    }
    return sortOrder === 'asc' ? comparison : -comparison;
  });


  // --- JSX Rendering ---
  return (
    <div style={{
      display: 'flex',
      minHeight: '100vh', // Use minHeight instead of height for flexibility
      backgroundColor: '#1a1a2e',
      color: 'white',
      // Adjust main layout based on screen size
      flexDirection: isMobile ? 'column' : 'row'
    }}>

      {/* Mobile Filter/Sort Toggle Button */}
      {isMobile && (
        <button
          onClick={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
          style={{
            position: 'fixed', // Or 'absolute' if preferred
            top: '560px',
            right: '15px',
            zIndex: 1050, // Ensure it's above other content
            background: '#16213e',
            border: '1px solid #0f3460',
            color: 'white',
            padding: '8px',
            borderRadius: '50%',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 5px rgba(0,0,0,0.3)'
          }}
          aria-label="Toggle sorting options"
        >
          <Filter size={20} />
        </button>
      )}

      {/* Sidebar - Sorting */}
      {/* Conditional rendering/styling for sidebar */}
      {(!isMobile || isMobileSidebarOpen) && (
        <div
          className="favorites-sidebar"
          style={{
            // --- Responsive Styles ---
            width: isMobile ? '80%' : '250px', // Takes more width on mobile when open
            maxWidth: isMobile ? '300px' : '250px', // Max width on mobile
            height: isMobile ? '100vh' : 'auto', // Full height on mobile overlay
            position: isMobile ? 'fixed' : 'relative', // Fixed position for mobile overlay
            top: 0,
            left: 0,
            zIndex: isMobile ? 1040 : 'auto', // Above content when mobile overlay
            transition: 'transform 0.3s ease-in-out', // Smooth transition for mobile
            transform: isMobile && !isMobileSidebarOpen ? 'translateX(-100%)' : 'translateX(0)', // Slide in/out
            // --- Original Styles (with adjustments) ---
            backgroundColor: '#16213e',
            padding: '20px',
            paddingTop: isMobile ? '50px' : '20px', // Add padding top for mobile close button
            display: 'flex',
            flexDirection: 'column',
            boxShadow: isMobile ? '5px 0px 15px rgba(0, 0, 0, 0.2)' : 'none', // Shadow for mobile overlay
            overflowY: isMobile ? 'auto' : 'visible' // Allow scrolling on mobile if needed
          }}
        >
           {/* Mobile Close Button */}
            {isMobile && (
                <button
                    onClick={() => setIsMobileSidebarOpen(false)}
                    style={{
                        position: 'absolute',
                        top: '10px',
                        right: '10px',
                        background: 'none',
                        border: 'none',
                        color: 'white',
                        cursor: 'pointer',
                    }}
                    aria-label="Close sorting options"
                >
                    <X size={24} />
                </button>
            )}

          <h3 style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
            <Filter style={{ marginRight: '10px' }} /> Sorting
          </h3>

          {/* Sorting Options */}
          <div>
            <h4 style={{ marginBottom: '10px'}}>Sort By</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' /* Add gap */ }}>
              {[
                { criteria: 'title', label: 'Title' },
                { criteria: 'dateLiked', label: 'Date Liked' }
              ].map(({ criteria, label }) => (
                <button
                  key={criteria}
                  onClick={() => sortLikedBooks(criteria)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between', // Pushes icon to the right
                    width: '100%', // Ensure full width
                    background: sortCriteria === criteria ? '#0f3460' : 'transparent',
                    border: `1px solid ${sortCriteria === criteria ? '#0f3460' : '#3a3a5e'}`, // Subtle border
                    color: 'white',
                    padding: '10px 15px', // Adjust padding
                    borderRadius: '5px',
                    marginBottom: '5px',
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontSize: '1rem' // Ensure readable font size
                  }}
                >
                  <span>{label}</span> {/* Wrap label for better control */}
                  {sortCriteria === criteria && (
                    sortOrder === 'asc'
                      ? <ArrowDownWideNarrow size={18} />
                      : <ArrowUp01 size={18} />
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}


       {/* Optional: Background Overlay for Mobile Sidebar */}
       {isMobile && isMobileSidebarOpen && (
           <div
               onClick={() => setIsMobileSidebarOpen(false)} // Close sidebar on overlay click
               style={{
                   position: 'fixed',
                   top: 0,
                   left: 0,
                   width: '100%',
                   height: '100%',
                   backgroundColor: 'rgba(0, 0, 0, 0.5)', // Semi-transparent black
                   zIndex: 1030, // Below sidebar but above content
                   backdropFilter: 'blur(2px)' // Optional blur effect
               }}
           />
       )}


      {/* Main Liked Books Content */}
      <div
        className="favorites-content"
        style={{
          flex: 1, // Takes remaining space
          // Remove fixed width/margin adjustments based on sidebar visibility on desktop if sidebar is always relative
          // On mobile, it naturally takes full width due to flex-direction: column
          backgroundColor: '#0f3460',
          padding: '20px',
          paddingTop: isMobile ? '60px' : '20px', // Add padding top on mobile if filter button is fixed
          overflowY: 'auto',
          height: isMobile ? 'auto' : '100vh' // Allow content to determine height on mobile, full viewport height on desktop
        }}
      >
        <h2 style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
          <Heart style={{ marginRight: '10px', fill: 'red', color: 'red' }} />
          Liked Books
        </h2>

        {/* --- Book List (Keep the existing logic) --- */}
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
          // Use a div or ul for the list container for better structure
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
             {processedLikedBooks.map((book) => (
                <div
                    key={book.google_book_id} // Use the guaranteed ID
                    style={{
                    display: 'flex',
                    // Make items stack vertically on very small screens if needed (optional)
                    // flexDirection: useMediaQuery('(max-width: 400px)') ? 'column' : 'row',
                    alignItems: 'center', // Or 'flex-start' if stacking vertically
                    backgroundColor: '#16213e',
                    borderRadius: '10px',
                    padding: '15px',
                    // marginBottom: '10px', // Replaced by gap in parent
                    transition: 'all 0.3s ease',
                    gap: '15px' // Add gap between image and text/button
                    }}
                    className="liked-book-item"
                >
                    <img
                        src={book.image_link || 'https://via.placeholder.com/80x120?text=No+Image'}
                        alt={book.title}
                        style={{
                            width: '80px',
                            height: '120px',
                            objectFit: 'cover',
                            // marginRight: '15px', // Replaced by gap in parent
                            borderRadius: '5px',
                            flexShrink: 0 // Prevent image from shrinking
                        }}
                        onError={(e) => { e.target.src = 'https://via.placeholder.com/80x120?text=Error'; }}
                    />

                    <div style={{ flex: 1, minWidth: 0 /* Prevent text overflow issues */ }}>
                        <h3 style={{ margin: 0, fontSize: '1.1rem', /* Slightly larger */ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' /* Prevent title wrap */ }}>{book.title}</h3>
                        <p style={{ margin: '5px 0', color: '#aaa', fontSize: '0.9rem' }}>{Array.isArray(book.authors) ? book.authors.join(', ') : book.authors}</p> {/* Handle array authors */}

                        {book.dateLiked instanceof Date && !isNaN(book.dateLiked) && (
                            <p style={{ margin: '5px 0', fontSize: '0.8rem', color: '#777' }}>
                            Liked on: {book.dateLiked.toLocaleDateString()}
                            </p>
                        )}
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center' }}>
                        <button
                            onClick={() => removeFromLikedBooks(book)}
                            style={{
                            background: 'none',
                            border: 'none',
                            color: 'white',
                            cursor: 'pointer',
                            padding: '5px' // Add some padding for easier clicking
                            }}
                            aria-label={`Remove ${book.title} from liked books`}
                        >
                            <Trash2 color="#ff6b6b" /> {/* Use a slightly different red */}
                        </button>
                    </div>
                </div>
             ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Liked;