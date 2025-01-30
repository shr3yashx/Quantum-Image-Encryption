from flask import Flask, request, render_template, send_file
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def quantum_xor(image, key):
    return np.bitwise_xor(image, key)

def generate_key(image_shape):
    return np.random.randint(0, 256, size=image_shape, dtype=np.uint8)

def encrypt_image(image):
    original_shape = image.shape
    side_length = min(original_shape[:2])
    image = cv2.resize(image, (side_length, side_length))
    key = generate_key(image.shape)
    encrypted = quantum_xor(image, key)
    return encrypted, key, original_shape

def decrypt_image(encrypted, key, original_shape):
    decrypted = quantum_xor(encrypted, key)
    decrypted = cv2.resize(decrypted, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return decrypted

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    encrypted_image, key, original_shape = encrypt_image(image)
    encrypted_path = os.path.join(app.config['UPLOAD_FOLDER'], 'encrypted_image.png')
    cv2.imwrite(encrypted_path, cv2.cvtColor(encrypted_image, cv2.COLOR_RGB2BGR))

    decrypted_image = decrypt_image(encrypted_image, key, original_shape)
    decrypted_path = os.path.join(app.config['UPLOAD_FOLDER'], 'decrypted_image.png')
    cv2.imwrite(decrypted_path, cv2.cvtColor(decrypted_image, cv2.COLOR_RGB2BGR))

    return render_template('result.html', encrypted_image='encrypted_image.png', decrypted_image='decrypted_image.png')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)
