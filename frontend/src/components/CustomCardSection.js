import React, { useState, useEffect } from 'react';
import { Heart, ListPlus } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './CustomCardSection.css';

const CustomCardSection = () => {
  const navigate = useNavigate();
  const [books, setBooks] = useState([]);
  const [likedBooks, setLikedBooks] = useState({});
  const [playlistBooks, setPlaylistBooks] = useState({});
  const [playlist, setPlaylist] = useState([]);
  const [liked, setLiked] = useState([]);

  // Retrieve logged in user (assumes user info is stored in localStorage)
  const getUserId = () => {
    const user = JSON.parse(localStorage.getItem('user'));
    return user ? user.id : null;
  };

  // Fetch user's playlist on load and update local state mapping
  const updatePlaylistState = async () => {
    const userId = getUserId();
    if (userId) {
      try {
        const response = await fetch(`http://localhost:5000/user/${userId}/playlist`, {
          method: "GET",
          headers: { "Content-Type": "application/json" }
        });
        const data = await response.json();
        const newPlaylistMap = {};
        data.forEach((book) => {
          newPlaylistMap[book.google_book_id] = true;
        });
        setPlaylistBooks(newPlaylistMap);
      } catch (error) {
        console.error("Error fetching playlist:", error);
      }
    }
  };

  // Fetch user's liked books and update local state mapping
  const updateLikedState = async () => {
    const userId = getUserId();
    if (userId) {
      try {
        const response = await fetch(`http://localhost:5000/user/${userId}/liked`, {
          method: "GET",
          headers: { "Content-Type": "application/json" }
        });
        const data = await response.json();
        const newLikedMap = {};
        data.forEach((book) => {
          newLikedMap[book.google_book_id] = true;
        });
        setLikedBooks(newLikedMap);
      } catch (error) {
        console.error("Error fetching liked books:", error);
      }
    }
  };

  useEffect(() => {
    updatePlaylistState();
    updateLikedState();
  }, []);

  // Fetch books from backend
  //changed info here to call the test function 
  useEffect(() => {
    const userId = getUserId()
    fetch(`http://localhost:5000/user/${userId}/info`, {
      method: "GET",
      headers: { "Content-Type": "application/json" }
    })
      .then((res) => res.json())
      .then((data) => {
        setBooks(data);
      })
      .catch((error) => console.error("Error fetching data:", error));
  }, []);

  // Fetch user's playlist on load (for local state)
  useEffect(() => {
    const userId = getUserId();
    if (userId) {
      fetch(`http://localhost:5000/user/${userId}/playlist`, {
        method: "GET",
        headers: { "Content-Type": "application/json" }
      })
        .then((res) => res.json())
        .then((data) => {
          if (Array.isArray(data)) {
            setPlaylist(data);
          }
        })
        .catch((error) => console.error("Error fetching playlist:", error));
    }
  }, []);

  // Fetch user's liked books on load (for local state)
  useEffect(() => {
    const userId = getUserId();
    if (userId) {
      fetch(`http://localhost:5000/user/${userId}/liked`, {
        method: "GET",
        headers: { "Content-Type": "application/json" }
      })
        .then((res) => res.json())
        .then((data) => {
          if (Array.isArray(data)) {
            setLiked(data);
          }
        })
        .catch((error) => console.error("Error fetching liked books:", error));
    }
  }, []);

  // Add book to playlist with backend integration
  const addToPlaylist = async (book) => {
    if (!playlist.some(item => item.google_book_id === (book.id || book.google_book_id))) {
      const userId = getUserId();
      if (!userId) {
        alert("Please log in to add books to your playlist.");
        return;
      }
      try {
        const response = await fetch(`http://localhost:5000/user/${userId}/playlist`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            google_book_id: book.id || book.google_book_id,
            title: book.title,
            authors: book.authors,
            synopsis: book.synopsis,
            rating: book.rating,
            genre: book.genre,
            image_link: book.image_link
          })
        });
        const data = await response.json();
        if (data.status === 'success') {
          setPlaylist([...playlist, book]);
        } else {
          alert(data.message);
        }
      } catch (err) {
        console.error(err);
        alert("Error adding book to playlist.");
      }
    }
  };

  // Add book to liked with backend integration
  const addToLiked = async (book) => {
    if (!liked.some(item => item.google_book_id === (book.id || book.google_book_id))) {
      const userId = getUserId();
      if (!userId) {
        alert("Please log in to like books.");
        return;
      }
      try {
        const response = await fetch(`http://localhost:5000/user/${userId}/liked`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            google_book_id: book.id || book.google_book_id,
            title: book.title,
            authors: book.authors,
            synopsis: book.synopsis,
            rating: book.rating,
            genre: book.genre,
            image_link: book.image_link
          })
        });
        const data = await response.json();
        if (data.status === 'success') {
          setLiked([...liked, book]);
        } else {
          alert(data.message);
        }
      } catch (err) {
        console.error(err);
        alert("Error adding book to liked.");
      }
    }
  };

  // Remove book from playlist with backend integration
  const removeFromPlaylist = async (bookToRemove) => {
    const userId = getUserId();
    if (!userId) {
      alert("Please log in to manage your playlist.");
      return;
    }
    
    try {
      const bookId = bookToRemove.id || bookToRemove.google_book_id;
      const response = await fetch(`http://localhost:5000/user/${userId}/playlist/${bookId}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" }
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setPlaylist(playlist.filter(book => 
          (book.id || book.google_book_id) !== (bookToRemove.id || bookToRemove.google_book_id)
        ));
      } else {
        alert(data.message);
      }
    } catch (err) {
      console.error(err);
      alert("Error removing book from playlist.");
    }
  };

  // Remove book from liked with backend integration
  const removeFromLiked = async (bookToRemove) => {
    const userId = getUserId();
    if (!userId) {
      alert("Please log in to manage your liked books.");
      return;
    }
    
    try {
      const bookId = bookToRemove.id || bookToRemove.google_book_id;
      const response = await fetch(`http://localhost:5000/user/${userId}/liked/${bookId}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" }
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setLiked(liked.filter(book => 
          (book.id || book.google_book_id) !== (bookToRemove.id || bookToRemove.google_book_id)
        ));
      } else {
        alert(data.message);
      }
    } catch (err) {
      console.error(err);
      alert("Error removing book from liked.");
    }
  };

  // Combined toggle function that updates playlist from backend
  const togglePlaylist = async (book) => {
    const userId = getUserId();
    if (!userId) {
      alert("Please log in to manage your playlist.");
      return;
    }
    
    if (playlistBooks[book.google_book_id]) {
      await removeFromPlaylist(book);
    } else {
      await addToPlaylist(book);
    }
    updatePlaylistState();
  };

  // Combined toggle function that updates liked from backend
  const toggleLiked = async (book) => {
    const userId = getUserId();
    if (!userId) {
      alert("Please log in to like books.");
      return;
    }
    
    if (likedBooks[book.google_book_id]) {
      await removeFromLiked(book);
    } else {
      await addToLiked(book);
    }
    updateLikedState();
  };

  return (
    <div>
      <h2 className="custom-font-weight-bold text-white">Recommended for you</h2>
      <div className="custom-card-container text-white">
        {books.map((book, index) => (
          <div 
            className="custom-card" 
            key={index}
            onClick={() => navigate(`/book/${book.google_book_id}`)}
            style={{ cursor: 'pointer' }}
          >
            <img 
              src={book.image_link} 
              alt={book.title} 
              onError={(e) => {
                e.target.onerror = null; 
                e.target.src = 'https://via.placeholder.com/250x150'
              }}
            />
            <div 
              className='d-flex justify-content-end' 
              style={{ gap: "10px" }}
              onClick={(e) => e.stopPropagation()} // Prevent navigation when clicking buttons
            >
              <button 
                onClick={() => toggleLiked(book)} 
                className='text-white bg-transparent border-0'
              >
                <Heart 
                  color={likedBooks[book.google_book_id] ? 'red' : 'white'} 
                  fill={likedBooks[book.google_book_id] ? 'red' : 'none'}
                  strokeWidth={2}
                />
              </button>
              <button className='text-white bg-transparent border-0'>
                <ListPlus 
                  onClick={() => togglePlaylist(book)}
                  color={playlistBooks[book.google_book_id] ? 'green' : 'white'} 
                  fill={playlistBooks[book.google_book_id] ? 'green' : 'none'}
                  strokeWidth={2}
                />
              </button>
            </div>
            <h3 className='text-white'>{book.title}</h3>
            <p className='text-white'>
              <strong>Authors:</strong> {book.authors}<br />
              <strong>Rating:</strong> {book.rating || 'N/A'}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CustomCardSection;