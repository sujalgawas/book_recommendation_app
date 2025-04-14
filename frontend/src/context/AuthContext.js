// src/context/AuthContext.js
import React, { createContext, useState, useEffect, useCallback, useContext } from 'react';
import { useNavigate } from 'react-router-dom'; // Use hook here if needed for context logic

// Create the context
const AuthContext = createContext(null);

// Create a provider component
export const AuthProvider = ({ children }) => {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [userData, setUserData] = useState(null);
    const [isLoading, setIsLoading] = useState(true); // Add loading state
    // Note: Don't use useNavigate directly here unless absolutely necessary for provider logic
    // It's better to let components consuming the context handle navigation

    // Function to safely get user data from localStorage
    const getUserDataFromStorage = useCallback(() => {
        try {
            const userString = localStorage.getItem('user');
            return userString ? JSON.parse(userString) : null;
        } catch (error) {
            console.error("Error parsing user data from localStorage:", error);
            localStorage.removeItem('user'); // Clear potentially corrupted data
            return null;
        }
    }, []);

    // Initialize auth state from localStorage on initial load
    useEffect(() => {
        const user = getUserDataFromStorage();
        if (user) {
            setIsLoggedIn(true);
            setUserData(user);
        }
        setIsLoading(false); // Finished loading initial auth state
        // Optional: Add storage event listener for cross-tab sync
        // window.addEventListener('storage', handleStorageChange);
        // return () => window.removeEventListener('storage', handleStorageChange);
    }, [getUserDataFromStorage]);

    // Login function - To be called from your Login component
    const login = (user) => {
        try {
            localStorage.setItem('user', JSON.stringify(user));
            setUserData(user);
            setIsLoggedIn(true);
            // Navigation should happen in the component calling login
        } catch (error) {
            console.error("Error saving user data to localStorage:", error);
            // Handle potential storage errors (e.g., quota exceeded)
        }
    };

    // Logout function - Can be called from anywhere (like Navbar)
    const logout = () => {
        localStorage.removeItem('user');
        setUserData(null);
        setIsLoggedIn(false);
        // Navigation should happen in the component calling logout (e.g., Navbar)
    };

    // Value provided to consuming components
    const value = {
        isLoggedIn,
        userData,
        isLoading, // Provide loading state if components need it
        login,
        logout,
    };

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook to easily consume the context
export const useAuth = () => {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    // Return null if context is explicitly set to null (optional, based on createContext default)
    if (context === null) {
         console.warn('AuthContext is null. Ensure AuthProvider wraps your component tree.');
         // Return a default structure or handle appropriately
         return { isLoggedIn: false, userData: null, isLoading: true, login: () => {}, logout: () => {} };
    }
    return context;
};