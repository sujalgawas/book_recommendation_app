// src/App.js
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Navbar from './components/navbar';
import Books from './components/books';
import Home from './components/Home';
import Liked from './components/liked';
import LoginPage from './components/login';
import BookPage from './components/BookPage';
import SearchResultsPage from './components/SearchResultsPage';
import GenrePreferencesPage from './components/GenrePreferencesPage'; 
import ProtectedRoute from './components/ProtectedRoute';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import { AuthProvider } from './context/AuthContext';

const App = () => {
  return (
    <div>
      <AuthProvider>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/books" element={<Books />} />
          <Route path="/liked" element={<Liked />} />
          <Route path="/login" element={<LoginPage/>}/>
          <Route path="/preferences" element={
            <ProtectedRoute>
              <GenrePreferencesPage />
            </ProtectedRoute>
          } />
          <Route path="/book/:id" element={<BookPage />} />
          <Route path="/search-results" element={<SearchResultsPage />} />
        </Routes>
      </AuthProvider>
    </div>
  );
};

export default App;