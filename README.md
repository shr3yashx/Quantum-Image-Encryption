# Quantum Image Encryption and Decryption

A Flask-based web application that demonstrates simple image encryption and decryption using XOR operations with a randomly generated key. This project resizes the image to a square for encryption and restores its original shape after decryption.

## Features
- Upload an image for encryption.
- View the encrypted image.
- Decrypt the encrypted image and view the result.
- Simple and user-friendly interface.

## Project Structure

project_directory/
├── app.py                   # Main Flask application
├── uploads/                 # Folder for uploaded and processed images
├── templates/               # HTML templates for the Flask app
│   ├── index.html           # Upload page template
│   └── result.html          # Results page template
├── static/                  # Static files (CSS, JS, etc.)
│   └── css/
│       └── style.css        # Styling for the web app
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
```

## Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/repo-name.git
   cd repo-name
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Flask app:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```
3. Upload an image, view the encrypted image, and the decrypted image on the results page.

## Dependencies
- Flask
- OpenCV (cv2)
- NumPy
- Werkzeug

Install dependencies using:
```bash
pip install -r requirements.txt
```

## How It Works
1. **Encryption**:
   - Resizes the image to a square shape.
   - Generates a random key of the same shape as the image.
   - Applies XOR operation to encrypt the image.
   
2. **Decryption**:
   - Applies the XOR operation with the same key to decrypt the image.
   - Restores the original shape of the image.

## Contributing
Feel free to open issues or submit pull requests for new features or improvements.
