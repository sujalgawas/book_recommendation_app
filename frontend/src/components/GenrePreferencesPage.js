// src/components/GenrePreferencesPage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Using axios for consistency as it was already used
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const GenrePreferencesPage = () => {
    const [selectedGenres, setSelectedGenres] = useState(new Set());
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();
    const { isLoggedIn, userData, isLoading: authLoading } = useAuth();

    // ... (useEffect for redirect, ALL_GENRES, handleGenreToggle remain the same) ...
     useEffect(() => {
         if (!authLoading) {
             if (!isLoggedIn || !userData) {
                 console.log("Not logged in, redirecting to login page");
                 navigate('/login');
             } else {
                 console.log("User is logged in, user ID:", userData.id);
             }
         }
     }, [isLoggedIn, userData, navigate, authLoading]);

     const ALL_GENRES = [
        'Fiction', 'Non-Fiction', 'Science Fiction', 'Fantasy', 'Mystery',
        'Thriller', 'Romance', 'Historical Fiction', 'Horror', 'Biography',
        'Self-Help', 'Young Adult', 'Children', 'Poetry', 'Comics & Graphic Novels',
        'Contemporary', 'Dystopian', 'Adventure', 'Paranormal', 'Humor'
    ];

     const handleGenreToggle = (genre) => {
         setError('');
         setSelectedGenres(prevSelected => {
             const newSelected = new Set(prevSelected);
             if (newSelected.has(genre)) {
                 newSelected.delete(genre);
             } else {
                 newSelected.add(genre);
             }
             return newSelected;
         });
     };

    const handleSubmit = async () => {
        if (selectedGenres.size === 0) {
            setError('Please select at least one genre.');
            return;
        }

        if (!userData || !userData.id) {
            setError('User information is missing. Please try logging in again.');
            return;
        }

        setIsLoading(true);
        setError('');
        const userId = userData.id; // Get user ID

        try {
            const genresArray = Array.from(selectedGenres);
            console.log("Selected genres:", genresArray);

            // --- Step 1: Add sample books to playlist based on preferences ---
            let booksAdded = 0;
            for (const genre of genresArray.slice(0, 3)) { // Use up to 3 genres
                try {
                    console.log(`Searching for books in genre: ${genre}`);
                    // Using axios.get as in the original code
                    const searchResponse = await axios.get(`http://localhost:5000/search?query=${genre}&num_results=3`);

                    if (searchResponse.data && searchResponse.data.length > 0) {
                        // Add up to 1 book per genre to user's playlist
                        for (const book of searchResponse.data.slice(0, 1)) {
                            console.log(`Adding book to playlist: ${book.title}`);
                            // Using axios.post as in the original code
                             await axios.post(`http://localhost:5000/user/${userId}/playlist_genre`, {
                                google_book_id: book.google_book_id,
                                title: book.title,
                                authors: book.authors,
                                genre: book.genre,
                                synopsis: book.synopsis,
                                rating: book.rating,
                                image_link: book.image_link,
                                tag: 'preference' // Keep the tag if your backend uses it
                            });
                            booksAdded++;
                        }
                    }
                } catch (err) {
                    console.error(`Error adding sample books for genre ${genre}:`, err);
                    // Decide if you want to stop or continue if adding a book fails
                    // setError(`Failed to add sample books for ${genre}. Continuing...`);
                    // Or throw new Error(`Failed to add books for ${genre}`); to stop
                }
            }

            if (booksAdded === 0) {
                 console.warn("No sample books were added to the playlist. Skipping recommendation generation.");
                 // Optionally inform the user or just navigate
                 // setError("Could not find sample books for selected genres.");
                 // setIsLoading(false); // Stop loading
                 // return; // Prevent further steps
            } else {
                console.log(`${booksAdded} sample book(s) added to the playlist.`);

                 // --- Step 2: Trigger recommendation generation ---
                 console.log("Triggering recommendation generation...");
                 const recResponse = await axios.post(`http://localhost:5000/user/${userId}/generate-recommendations`);

                 // Optional: Check response status if needed
                 if (recResponse.data?.status !== 'success') {
                     // Handle failure - maybe log, show error, don't clear playlist?
                     console.error("Recommendation generation failed:", recResponse.data?.message);
                     throw new Error(recResponse.data?.message || 'Failed to generate recommendations.');
                 }
                 console.log("Recommendation generation successful.");


                 // --- Step 3: Clear the user's playlist ---
                 console.log("Clearing user's playlist...");
                 const clearResponse = await axios.delete(`http://localhost:5000/user/${userId}/clear-playlist`);

                  // Optional: Check response status if needed
                 if (clearResponse.data?.status !== 'success') {
                     // Handle failure - maybe log, show error? The recommendations were generated anyway.
                     console.error("Playlist clearing failed:", clearResponse.data?.message);
                     // Don't throw an error here maybe, as the main goal (recs) was achieved
                     setError("Recommendations generated, but failed to clear temporary playlist.");
                 } else {
                    console.log("User playlist cleared successfully.");
                 }
            }


            // --- Step 4: Navigate to homepage ---
            console.log("Preferences processed, navigating to homepage");
            navigate('/'); // Navigate after all steps (or if book adding failed but you decide to continue)

        } catch (err) {
            console.error("Error processing genre preferences:", err);
            // More specific error message if possible
            setError(err.message || err.response?.data?.message || 'An error occurred. Please try again.');
        } finally {
            setIsLoading(false); // Ensure loading state is always turned off
        }
    };

    // --- Styling objects (pageStyle, gridStyle, etc.) remain the same ---
    const pageStyle = {
        padding: '40px 20px',
        maxWidth: '900px',
        margin: 'auto',
        textAlign: 'center',
        color: '#e0e0e0',
        backgroundColor: '#121212',
        minHeight: 'calc(100vh - 56px)',
    };

    const gridStyle = {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
        gap: '15px',
        marginTop: '30px',
        marginBottom: '40px',
    };

    const genreButtonStyle = {
        padding: '12px 18px',
        borderRadius: '25px',
        border: '2px solid #444',
        background: '#282828',
        color: '#ccc',
        cursor: 'pointer',
        transition: 'all 0.2s ease-in-out',
        fontSize: '0.95rem',
        textAlign: 'center',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
    };

    const selectedGenreStyle = {
        ...genreButtonStyle,
        background: '#007BFF',
        borderColor: '#007BFF',
        color: '#fff',
        fontWeight: 'bold',
        boxShadow: '0 0 10px rgba(0, 123, 255, 0.5)',
    };

    const submitButtonStyle = {
        padding: '14px 35px',
        borderRadius: '8px',
        border: 'none',
        backgroundColor: '#007BFF',
        color: '#fff',
        cursor: isLoading || selectedGenres.size === 0 ? 'not-allowed' : 'pointer',
        fontSize: '1.1rem',
        fontWeight: 'bold',
        opacity: isLoading || selectedGenres.size === 0 ? 0.6 : 1,
        transition: 'all 0.2s ease',
        marginTop: '10px',
    };

    const errorStyle = {
        color: '#ff4d4d',
        marginTop: '20px',
        minHeight: '1.2em',
        fontWeight: 'bold',
    };

    // --- Render logic remains the same ---
     if (authLoading) {
         return (
             <div style={pageStyle}>
                 <h2 style={{ color: '#fff' }}>Loading...</h2>
             </div>
         );
     }

    return (
        <div style={pageStyle}>
             <h2 style={{ color: '#fff', marginBottom: '10px' }}>Welcome to BookRecs!</h2>
             <p style={{ fontSize: '1.1rem', marginBottom: '30px' }}>
                 To help us personalize your experience, please select your favorite book genres:
             </p>

             <div style={gridStyle}>
                 {ALL_GENRES.map(genre => (
                     <button
                         key={genre}
                         style={selectedGenres.has(genre) ? selectedGenreStyle : genreButtonStyle}
                         onClick={() => handleGenreToggle(genre)}
                         onMouseOver={(e) => {
                             if (!selectedGenres.has(genre)) {
                                 e.currentTarget.style.backgroundColor = '#383838';
                                 e.currentTarget.style.borderColor = '#666';
                             }
                         }}
                         onMouseOut={(e) => {
                             if (!selectedGenres.has(genre)) {
                                 e.currentTarget.style.backgroundColor = '#282828';
                                 e.currentTarget.style.borderColor = '#444';
                             }
                         }}
                     >
                         {genre}
                     </button>
                 ))}
             </div>

             {error && <p style={errorStyle}>{error}</p>}

             <button
                 style={submitButtonStyle}
                 onClick={handleSubmit}
                 disabled={isLoading || selectedGenres.size === 0}
                 onMouseOver={(e) => {
                     if (!(isLoading || selectedGenres.size === 0)) {
                         e.currentTarget.style.backgroundColor = '#0056b3';
                     }
                 }}
                 onMouseOut={(e) => {
                     if (!(isLoading || selectedGenres.size === 0)) {
                         e.currentTarget.style.backgroundColor = '#007BFF';
                     }
                 }}
             >
                 {isLoading ? 'Processing...' : 'Get Started with Recommendations'}
             </button>
        </div>
    );
};

export default GenrePreferencesPage;