from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image

app = Flask(__name__)
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

@app.route("/ocr", methods=["POST"])
def ocr_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file.stream).convert("RGB")

    result = ocr.ocr(image, cls=True)
    texts = [line[1][0] for line in result[0]] if result and result[0] else []
    return jsonify({"text": texts})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
