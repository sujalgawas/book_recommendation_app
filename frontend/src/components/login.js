// LoginPage.js - Updated to work with your AuthContext
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const LoginPage = () => {
    const [isLogin, setIsLogin] = useState(true);
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();
    const { login, isLoggedIn, userData } = useAuth();

    // Redirect if already logged in
    useEffect(() => {
        if (isLoggedIn && userData) {
            navigate('/');
        }
    }, [isLoggedIn, userData, navigate]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        try {
            if (isLogin) {
                // --- Login Logic ---
                const response = await axios.post('http://localhost:5000/login', { username, password });
                console.log("Login response:", response.data);

                if (response.data.status === 'success' && response.data.user) {
                    // Store user data in context
                    login(response.data.user);
                    
                    // Check if user has preferences already
                    try {
                        const userInfo = await axios.get(`http://localhost:5000/user/${response.data.user.id}/info`);
                        console.log("User info response:", userInfo.data);
                        
                        // If user has no recommendations yet, redirect to preferences page
                        if (!userInfo.data || 
                            !userInfo.data.playlists || 
                            userInfo.data.playlists.length === 0) {
                            console.log("Redirecting to preferences");
                            navigate('/');
                        } else {
                            // User already has recommendations, go to home
                            console.log("Redirecting to home");
                            navigate('/');
                        }
                    } catch (err) {
                        console.error("Error checking user info:", err);
                        // If there's an error checking preferences, default to preferences page
                        navigate('/');
                    }
                } else {
                    setError(response.data.message || 'Login failed. Please check credentials.');
                }
            } else {
                // --- Signup Logic ---
                const response = await axios.post('http://localhost:5000/signup', { username, email, password });
                console.log("Signup response:", response.data);

                if (response.data.status === 'success') {
                    if (response.data.user) {
                        // Auto-login after signup
                        login(response.data.user);
                        console.log("Signed up and logged in, redirecting to preferences");
                        navigate('/preferences');
                    } else {
                        // If server doesn't return user object, switch to login
                        setIsLogin(true);
                        setUsername('');
                        setPassword('');
                        setEmail('');
                        setError('Signup successful! Please log in.');
                    }
                } else {
                    setError(response.data.message || 'Signup failed. Please try again.');
                }
            }
        } catch (err) {
            setError(err.response?.data?.message || 'An error occurred. Please try again.');
            console.error("Login/Signup Error:", err);
        }
    };

    // Styles remain the same as your original code
    const containerStyle = {
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: 'calc(100vh - 56px)',
        backgroundColor: '#121212',
        color: '#e0e0e0',
        padding: '20px'
    };

    const formStyle = {
        backgroundColor: '#1e1e1e',
        padding: '30px',
        borderRadius: '10px',
        boxShadow: '0 5px 15px rgba(0, 0, 0, 0.5)',
        width: '100%',
        maxWidth: '350px'
    };

    const inputStyle = {
        width: '100%',
        padding: '12px',
        margin: '10px 0',
        borderRadius: '5px',
        border: '1px solid #444',
        backgroundColor: '#333',
        color: '#fff',
        boxSizing: 'border-box'
    };

    const buttonStyle = {
        width: '100%',
        padding: '12px',
        marginTop: '15px',
        borderRadius: '5px',
        border: 'none',
        backgroundColor: '#007BFF',
        color: '#fff',
        cursor: 'pointer',
        fontSize: '1rem',
        fontWeight: 'bold',
        transition: 'background-color 0.2s ease'
    };

    const toggleStyle = {
        color: '#00A0FF',
        cursor: 'pointer',
        textDecoration: 'underline'
    };

    const errorStyle = {
       color: '#ff4d4d',
       textAlign: 'center',
       marginBottom: '15px',
       fontSize: '0.9rem'
    };

    const labelStyle = {
        display: 'block',
        marginBottom: '2px',
        fontWeight: 'bold',
        fontSize: '0.9rem'
    };


    return (
        <div style={containerStyle}>
            <form onSubmit={handleSubmit} style={formStyle}>
                <h2 style={{ textAlign: 'center', marginBottom: '25px', color: '#fff' }}>
                    {isLogin ? 'Login' : 'Sign Up'}
                </h2>
                {error && <p style={errorStyle}>{error}</p>}
                <div style={{ marginBottom: '15px' }}>
                    <label style={labelStyle}>Username:</label>
                    <input
                        style={inputStyle}
                        type="text"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        required
                        placeholder="Enter your username"
                    />
                </div>
                {!isLogin && (
                    <div style={{ marginBottom: '15px' }}>
                        <label style={labelStyle}>Email:</label>
                        <input
                            style={inputStyle}
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            placeholder="Enter your email"
                        />
                    </div>
                )}
                <div style={{ marginBottom: '15px' }}>
                    <label style={labelStyle}>Password:</label>
                    <input
                        style={inputStyle}
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                        placeholder="Enter your password"
                    />
                </div>
                <button type="submit" style={buttonStyle}
                   onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#0056b3'}
                   onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#007BFF'}
                >
                    {isLogin ? 'Login' : 'Sign Up'}
                </button>
                <p style={{ marginTop: '20px', textAlign: 'center', fontSize: '0.9rem' }}>
                    {isLogin ? "Don't have an account?" : "Already have an account?"}{' '}
                    <span
                        style={toggleStyle}
                        onClick={() => {
                            setIsLogin(!isLogin);
                            setUsername('');
                            setPassword('');
                            setEmail('');
                            setError('');
                        }}
                    >
                        {isLogin ? 'Sign Up' : 'Login'}
                    </span>
                </p>
            </form>
        </div>
    );
};

export default LoginPage;