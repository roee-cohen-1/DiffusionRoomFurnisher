from flask import Flask, render_template, request, jsonify
import base64
from PIL import Image
import io
from model import run


app = Flask('demo')

@app.route("/")
def index():
    return render_template('demo.html')


@app.route('/process', methods=['POST'])
def process_image():
    image_data = request.form['image_data']
    # Removing the 'data:image/png;base64,' part
    image_data = image_data.replace('data:image/jpeg;base64,', '')
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image.save('image.jpeg')
    image = run(image)
    # image = image.convert('L')  # for testing - change images to grayscale
    buffered = io.BytesIO()
    image.save(buffered, format='JPEG')
    image = base64.b64encode(buffered.getvalue()).decode()
    return jsonify({
        'image': f'data:image/jpeg;base64,{image}'
    })
