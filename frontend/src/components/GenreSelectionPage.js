import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext'; // Import useAuth to update context

// Configure axios instance (optional but recommended)
const apiClient = axios.create({
    baseURL: 'http://localhost:5000',
    withCredentials: true,
});

// --- Example Genres (Fetch from backend ideally) ---
const ALL_GENRES = [
    'Fiction', 'Non-Fiction', 'Science Fiction', 'Fantasy', 'Mystery',
    'Thriller', 'Romance', 'Historical Fiction', 'Horror', 'Biography',
    'Self-Help', 'Young Adult', 'Children', 'Poetry', 'Comics & Graphic Novels',
    'Contemporary', 'Dystopian', 'Adventure', 'Paranormal', 'Humor'
];

const GenreSelectionPage = () => {
    const [selectedGenres, setSelectedGenres] = useState(new Set());
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();
    const { login } = useAuth(); // Get login function to update context state after success

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
        if (selectedGenres.size < 1) {
             setError('Please select at least one genre to continue.');
             return;
        }

        setIsLoading(true);
        setError('');

        try {
            const genresArray = Array.from(selectedGenres);
            console.log("Submitting genres:", genresArray);

            // Assuming backend route is /api/user/preferences
            const response = await apiClient.post('/api/user/preferences', { genres: genresArray });
            console.log("API Response from /api/user/preferences:", response.data);

            // Check for success AND if the updated user object is returned by the backend
            if (response.data.status === 'success' && response.data.user) {
                console.log("Preferences saved successfully on backend.");
                // Update the AuthContext with the fresh user data (includes has_selected_preferences=true)
                login(response.data.user);
                console.log("AuthContext updated with new user data.");
                // Navigate to homepage after successful update
                navigate('/');
            } else {
                 // Handle cases where backend reported success but didn't return user data, or reported failure.
                 setError(response.data.message || 'Failed to save preferences or missing user data in response.');
                 console.error("Preference saving failed or missing user data:", response.data);
            }
        } catch (err) {
            console.error("Error saving preferences:", err.response || err);
            setError(err.response?.data?.message || 'An error occurred saving preferences.');
        } finally {
            setIsLoading(false);
        }
    };

    // --- Styles ---
     const pageStyle = {
        padding: '40px 20px',
        maxWidth: '900px',
        margin: 'auto',
        textAlign: 'center',
        color: '#e0e0e0',
        backgroundColor: '#121212',
        minHeight: 'calc(100vh - 56px - 40px)', // Adjust based on navbar height and padding
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

    // --- Render component ---
    return (
        <div style={pageStyle}>
            <h2 style={{ color: '#fff', marginBottom: '10px' }}>Welcome!</h2>
            <p style={{ fontSize: '1.1rem', marginBottom: '30px' }}>
                Help us personalize your experience. Select your favorite genres (choose at least one):
            </p>
            <div style={gridStyle}>
                {ALL_GENRES.map(genre => (
                    <button
                        key={genre}
                        style={selectedGenres.has(genre) ? selectedGenreStyle : genreButtonStyle}
                        onClick={() => handleGenreToggle(genre)}
                        onMouseOver={(e) => { if (!selectedGenres.has(genre)) e.currentTarget.style.backgroundColor = '#383838'; e.currentTarget.style.borderColor = '#666'; }}
                        onMouseOut={(e) => { if (!selectedGenres.has(genre)) e.currentTarget.style.backgroundColor = '#282828'; e.currentTarget.style.borderColor = '#444'; }}
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
            >
                {isLoading ? 'Saving...' : 'Continue to Recommendations'}
            </button>
        </div>
    );
};

export default GenreSelectionPage;