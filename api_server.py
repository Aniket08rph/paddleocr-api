from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import requests
from PIL import Image
from io import BytesIO
import numpy as np

app = Flask(__name__)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.route("/ocr", methods=["POST"])
def analyze_image():
    image_url = request.json.get("image")
    if not image_url:
        return jsonify({"error": "Image URL missing"}), 400

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    result = ocr.ocr(np.array(image))
    text = " ".join([line[1][0] for block in result for line in block])
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
