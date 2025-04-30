// src/components/Navbar.js
import React, { useState } from 'react'; // Removed useEffect, useCallback
import { NavLink, Link, useNavigate } from 'react-router-dom';
import { Search, User, LogOut } from 'lucide-react';
import { useAuth } from '../context/AuthContext'; // Adjust path if needed

const Navbar = () => {
    const navigate = useNavigate();
    const [searchQuery, setSearchQuery] = useState('');
    const { isLoggedIn, userData, logout, isLoading } = useAuth(); // Get state and functions from context

    // If still loading initial auth state, maybe show nothing or a loader
    if (isLoading) {
        return null; // Or return a simplified loading navbar
    }

    const handleSubmit = (e) => {
        e.preventDefault();
        const trimmedQuery = searchQuery.trim();
        if (trimmedQuery) {
            navigate(`/search-results?query=${encodeURIComponent(trimmedQuery)}`);
        } else {
            console.log("Search query is empty");
        }
    };

    const handleLogout = () => {
        logout(); // Call the logout function from context
        navigate('/login'); // Redirect after logout
    };

    // Placeholder profile picture or initials generator (no changes needed here)
    const getProfileDisplay = () => {
       if (userData?.profilePicUrl) {
            return <img src={userData.profilePicUrl} alt="Profile" width="32" height="32" className="rounded-circle" />;
        } else if (userData?.username) {
            const initials = userData.username.charAt(0).toUpperCase();
            return (
                <span className="d-inline-flex align-items-center justify-content-center rounded-circle bg-info text-dark" style={{ width: '32px', height: '32px', fontSize: '0.9rem', fontWeight: 'bold' }}>
                    {initials}
                </span>
            );
        } else {
             return <User size={24} />;
        }
    };

    return (
        <nav className="navbar navbar-expand-lg navbar-dark bg-dark sticky-top py-2">
            <div className="container-fluid">
                {/* Brand */}
                <Link className="navbar-brand fw-bold me-3" to="/">
                    BOOK REC
                </Link>

                {/* Navbar Toggler */}
                <button
                    className="navbar-toggler" type="button" data-bs-toggle="collapse"
                    data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent"
                    aria-expanded="false" aria-label="Toggle navigation"
                >
                    <span className="navbar-toggler-icon"></span>
                </button>

                {/* Collapsible Content */}
                <div className="collapse navbar-collapse" id="navbarSupportedContent">
                    {/* Search Form */}
                    <form
                        className="d-flex my-2 my-lg-0 me-auto ms-lg-3"
                        onSubmit={handleSubmit} role="search"
                    >
                        <input
                            className="form-control me-2" type="search" placeholder="Search Books..."
                            aria-label="Search" value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            style={{ minWidth: '250px', maxWidth: '400px' }}
                        />
                        <button className="btn btn-outline-info" type="submit">
                            <Search size={20} />
                        </button>
                    </form>

                    {/* Navigation Links */}
                    <ul className="navbar-nav mb-2 mb-lg-0 align-items-lg-center">
                       <li className="nav-item">
                             <NavLink className="nav-link" end to="/"> Home </NavLink>
                       </li>
                        {/* Show these links only if logged in (optional) */}
                        {isLoggedIn && (
                            <>
                                <li className="nav-item">
                                    <NavLink className="nav-link" to="/books"> My List </NavLink>
                                </li>
                                <li className="nav-item">
                                    <NavLink className="nav-link" to="/liked"> Liked Books </NavLink>
                                </li>
                                <li className="nav-item">
                                    <NavLink className="nav-link" to="/ask-gemini"> ChatBot </NavLink>
                                </li>
                            </>
                        )}
                    </ul>

                    {/* Profile/Login Dropdown */}
                    <div className="navbar-nav ms-auto align-items-lg-center">
                        <div className="nav-item dropdown">
                            <button
                                className="nav-link dropdown-toggle d-flex align-items-center"
                                id="navbarDropdownMenuLink" role="button" data-bs-toggle="dropdown"
                                aria-expanded="false"
                                style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,.55)' }}
                            >
                                {/* Use context state here */}
                                {isLoggedIn ? getProfileDisplay() : <User size={24} />}
                                {isLoggedIn && userData?.username && (
                                    <span className='d-none d-lg-inline ms-2'>{userData.username}</span>
                                )}
                            </button>
                            <ul
                                className="dropdown-menu dropdown-menu-end dropdown-menu-dark"
                                aria-labelledby="navbarDropdownMenuLink"
                            >
                                {isLoggedIn ? (
                                    // Logged In options
                                    <li>
                                        <button className="dropdown-item d-flex align-items-center gap-2" onClick={handleLogout}>
                                            <LogOut size={16} /> Logout
                                        </button>
                                    </li>
                                    // Add other items like 'Profile', 'Settings' here if needed
                                ) : (
                                    // Logged Out options
                                    <>
                                        <li><Link className="dropdown-item" to="/login">Login</Link></li>
                                    </>
                                )}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;