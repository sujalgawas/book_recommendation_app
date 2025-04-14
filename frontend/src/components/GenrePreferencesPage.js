// src/components/GenrePreferencesPage.js - Updated to work with your AuthContext
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const GenrePreferencesPage = () => {
    const [selectedGenres, setSelectedGenres] = useState(new Set());
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();
    const { isLoggedIn, userData, isLoading: authLoading } = useAuth();

    // Redirect if no user is logged in
    useEffect(() => {
        // Only check after auth state is loaded
        if (!authLoading) {
            if (!isLoggedIn || !userData) {
                console.log("Not logged in, redirecting to login page");
                navigate('/login');
            } else {
                console.log("User is logged in, user ID:", userData.id);
            }
        }
    }, [isLoggedIn, userData, navigate, authLoading]);

    // Genres list - can be expanded or fetched from backend if needed
    const ALL_GENRES = [
        'Fiction', 'Non-Fiction', 'Science Fiction', 'Fantasy', 'Mystery',
        'Thriller', 'Romance', 'Historical Fiction', 'Horror', 'Biography',
        'Self-Help', 'Young Adult', 'Children', 'Poetry', 'Comics & Graphic Novels',
        'Contemporary', 'Dystopian', 'Adventure', 'Paranormal', 'Humor'
    ];

    const handleGenreToggle = (genre) => {
        setError(''); // Clear error on interaction
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

        try {
            const genresArray = Array.from(selectedGenres);
            console.log("Submitting genres:", genresArray);
            
            // Create a few sample books from selected genres to add to recommendations
            for (const genre of genresArray.slice(0, 3)) { // Use up to 3 genres
                try {
                    // Search for books in this genre
                    console.log(`Searching for books in genre: ${genre}`);
                    const searchResponse = await axios.get(`http://localhost:5000/search?query=${genre}&num_results=3`);
                    
                    if (searchResponse.data && searchResponse.data.length > 0) {
                        // Add books to user's playlist to trigger recommendations
                        for (const book of searchResponse.data.slice(0, 1)) { // Add up to 1 books per genre
                            console.log(`Adding book to playlist: ${book.title}`);
                            await axios.post(`http://localhost:5000/user/${userData.id}/playlist_genre`, {
                                google_book_id: book.google_book_id,
                                title: book.title,
                                authors: book.authors,
                                genre: book.genre,
                                synopsis: book.synopsis,
                                rating: book.rating,
                                image_link: book.image_link,
                                tag: 'preference' // Special tag to identify preference-based books
                            });
                        }
                        const response = await fetch(`http://localhost:5000/user/${userData.id}/Ai`, {
                            method: "POST", headers: { "Content-Type": "application/json" }
                        });
                        const data = await response.json()
                        console.error(data)
                    }
                } catch (err) {
                    console.error(`Error adding sample books for genre ${genre}:`, err);
                    // Continue with other genres even if one fails
                }
            }

            console.log("Preferences saved successfully, navigating to homepage");
            // Navigate to homepage after processing genres
            navigate('/');
        } catch (err) {
            console.error("Error processing genre preferences:", err);
            setError(err.response?.data?.message || 'An error occurred. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    // --- Styling (matching the dark theme from your provided code) ---
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

    // If still loading auth, show loading indicator
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