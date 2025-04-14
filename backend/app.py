from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

#test for git push action 
app = Flask(__name__)

# Configuration for PostgreSQL database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost:5432/book'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Define a simple User model
class User(db.Model):
    __tablename__ = 'users'  # Table name in PostgreSQL
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<User {self.username}>'

# A simple route for testing
@app.route('/')
def index():
    return "Flask with PostgreSQL is working!"

if __name__ == '__main__':
    app.run(debug=True,port=5001)
