// src/components/ProtectedRoute.js
import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext'; // Adjust path if needed

const ProtectedRoute = ({ children }) => {
    const { isLoggedIn, userData, isLoading } = useAuth();
    const location = useLocation();

    // Optional Debug Log
    // console.log(
    //     `ProtectedRoute Evaluating: Path=${location.pathname}, Loading=${isLoading}, LoggedIn=${isLoggedIn}, UserData Exists=${!!userData}, PrefsSet=${userData?.has_selected_preferences}`
    // );

    // 1. Handle Loading State
    if (isLoading) {
        return <div>Loading...</div>;
    }

    // 2. Handle Not Logged In
    if (!isLoggedIn) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    // 3. Handle Missing User Data (Safety Check)
    if (!userData) {
        console.error("ProtectedRoute Error: Logged in but userData is missing! Redirecting to login.");
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    // 4. Specific logic for the '/preferences' route
    if (location.pathname === '/preferences') {
        if (userData.has_selected_preferences === true) {
            // Preferences ARE set, redirect away from this page
            return <Navigate to="/" replace />;
        } else {
            // Preferences are NOT set, allow access
            return children;
        }
    }

    // 5. Allow access to all OTHER protected routes
    return children;
};

export default ProtectedRoute;