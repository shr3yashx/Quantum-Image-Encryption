import numpy as np
from PIL import Image
import os
import random
from flask import Flask, request, render_template, send_from_directory

# Import Qiskit modules needed for visualization.
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define the uploads directory and ensure it exists.
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def generate_bb84_key_vectorized(key_length, eavesdrop_prob=0.0):
    sifted_alice_bits = []  # Accumulate the bits that Alice would have sent.
    sifted_bob_bits = []  # Accumulate Bob's measurement outcomes (the sifted key).

    # Continue until we have at least 'key_length' bits.
    while len(sifted_bob_bits) < key_length:
        # Use a safety margin (e.g. 3 times as many qubits as needed in this iteration)
        n_qubits = int((key_length - len(sifted_bob_bits)) * 3)

        # Generate random bits and bases for Alice and Bob.
        alice_bits = np.random.randint(2, size=n_qubits)
        alice_bases = np.random.randint(2, size=n_qubits)
        bob_bases = np.random.randint(2, size=n_qubits)
        eavesdrop_events = np.random.rand(n_qubits) < eavesdrop_prob

        bob_results = np.empty(n_qubits, dtype=int)

        # For qubits where bases do NOT match, Bob's result is completely random.
        mask_diff = (alice_bases != bob_bases)
        bob_results[mask_diff] = np.random.randint(2, size=np.count_nonzero(mask_diff))

        # For qubits where bases match:
        mask_same = (alice_bases == bob_bases)
        # Without eavesdropping, Bob's result equals Alice's bit.
        mask_same_no_eve = mask_same & (~eavesdrop_events)
        bob_results[mask_same_no_eve] = alice_bits[mask_same_no_eve]
        # With eavesdropping, Bob's bit is correct only 50% of the time.
        mask_same_eve = mask_same & eavesdrop_events
        num_eve = np.count_nonzero(mask_same_eve)
        if num_eve > 0:
            coin_flip = np.random.rand(num_eve) < 0.5
            bob_results[mask_same_eve] = np.where(coin_flip,
                                                  alice_bits[mask_same_eve],
                                                  1 - alice_bits[mask_same_eve])

        # Append only the bits where the bases matched.
        sifted_alice_bits.extend(alice_bits[mask_same])
        sifted_bob_bits.extend(bob_results[mask_same])

    # Convert to numpy arrays for error analysis.
    sifted_alice_bits = np.array(sifted_alice_bits)
    sifted_bob_bits = np.array(sifted_bob_bits)

    # Check the error rate in the sifted key.
    error_rate = np.mean(sifted_alice_bits != sifted_bob_bits)
    if error_rate > 0:
        raise Exception(f"Eavesdropper detected! Error rate: {error_rate:.2f}")

    # Use the first 'key_length' bits from Bob's sifted key.
    key_bits = sifted_bob_bits[:key_length].tolist()
    return key_bits


def encrypt_image(image_path, key_bits):
    # Open the image in RGB mode.
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    data = image.tobytes()

    key_str = ''.join(str(bit) for bit in key_bits)
    key_bytes = bytearray()
    for i in range(0, len(key_str), 8):
        byte = key_str[i:i + 8]
        if len(byte) < 8:
            byte = byte.ljust(8, '0')
        key_bytes.append(int(byte, 2))
    # Repeat key bytes if necessary.
    full_key = (key_bytes * ((len(data) // len(key_bytes)) + 1))[:len(data)]
    encrypted_data = bytes([b ^ k for b, k in zip(data, full_key)])

    enc_image_path = os.path.join(UPLOAD_FOLDER, "encrypted_image.png")
    enc_image = Image.frombytes('RGB', image.size, encrypted_data)
    enc_image.save(enc_image_path)
    print(f"Encrypted image saved at: {enc_image_path}")
    return enc_image_path


def decrypt_image(encrypted_path, key_bits):
    encrypted_image = Image.open(encrypted_path)
    if encrypted_image.mode != 'RGB':
        encrypted_image = encrypted_image.convert('RGB')
    encrypted_data = encrypted_image.tobytes()

    key_str = ''.join(str(bit) for bit in key_bits)
    key_bytes = bytearray()
    for i in range(0, len(key_str), 8):
        byte = key_str[i:i + 8]
        if len(byte) < 8:
            byte = byte.ljust(8, '0')
        key_bytes.append(int(byte, 2))
    full_key = (key_bytes * ((len(encrypted_data) // len(key_bytes)) + 1))[:len(encrypted_data)]
    decrypted_data = bytes([b ^ k for b, k in zip(encrypted_data, full_key)])

    dec_image_path = os.path.join(UPLOAD_FOLDER, "decrypted_image.png")
    dec_image = Image.frombytes('RGB', encrypted_image.size, decrypted_data)
    dec_image.save(dec_image_path)
    print(f"Decrypted image saved at: {dec_image_path}")
    return dec_image_path


def visualize_pixel_qubits(pixel_value, output_filename="visualization.png", basis="computational"):
    bit_str = format(pixel_value, '08b')
    print(f"Visualizing pixel value: {pixel_value} -> Bits: {bit_str}")

    fig, axs = plt.subplots(2, 4, subplot_kw={'projection': '3d'}, figsize=(12, 6))
    axs = axs.flatten()

    for idx, bit_char in enumerate(bit_str):
        qc = QuantumCircuit(1)
        if basis == "computational":
            if bit_char == '1':
                qc.x(0)
        else:
            raise ValueError("Unsupported basis. Use 'computational'.")
        state = Statevector.from_instruction(qc)
        a, b = state.data
        x = 2 * np.real(a * np.conjugate(b))
        y = 2 * np.imag(a * np.conjugate(b))
        z = np.abs(a) ** 2 - np.abs(b) ** 2
        bloch = [x, y, z]
        plot_bloch_vector(bloch, ax=axs[idx])
        axs[idx].set_title(f"Bit {idx}: {bit_char}")

    plt.tight_layout()
    out_path = os.path.join(UPLOAD_FOLDER, output_filename)
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Visualization saved at: {out_path}")
    return out_path


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form inputs.
            eavesdrop_prob = float(request.form.get('eavesdrop_prob', 0.0))
            visualization_x = int(request.form.get('visualization_x', 0))
            visualization_y = int(request.form.get('visualization_y', 0))

            # Save the uploaded image.
            image_file = request.files['file']
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)
            print(f"Image saved at: {image_path}")

            # Open image in RGB mode.
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_data = image.tobytes()
            n_bytes = len(image_data)
            required_key_length = 8 * n_bytes  # one bit per byte * 8

            # Generate key using vectorized simulation.
            key_bits = generate_bb84_key_vectorized(required_key_length, eavesdrop_prob=eavesdrop_prob)

            # Encrypt and decrypt the image.
            encrypted_image_path = encrypt_image(image_path, key_bits)
            decrypted_image_path = decrypt_image(encrypted_image_path, key_bits)

            # Open the encrypted image.
            enc_img = Image.open(encrypted_image_path)
            if enc_img.mode != 'RGB':
                enc_img = enc_img.convert('RGB')
            width, height = enc_img.size

            # Validate the user-specified coordinates.
            if visualization_x < 0 or visualization_x >= width or visualization_y < 0 or visualization_y >= height:
                return render_template('index.html',
                                       message=f"Invalid visualization coordinates. Image size is {width}x{height}.")

            # For visualization, we select one channel from the pixel. Here we choose the red channel.
            pixel_value = enc_img.getpixel((visualization_x, visualization_y))[0]
            visualization_path = visualize_pixel_qubits(pixel_value, output_filename="visualization.png")

            return render_template('result.html',
                                   encrypted_image=os.path.basename(encrypted_image_path),
                                   decrypted_image=os.path.basename(decrypted_image_path),
                                   visualization_image=os.path.basename(visualization_path))
        except Exception as e:
            return render_template('index.html', message=f"Error during image processing: {str(e)}")

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
