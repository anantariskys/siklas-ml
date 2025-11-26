from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.model_handler import predict
from utils.preprocessing import preprocess_text

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Selamat datang di API SiKlas"})

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()

    if not data or "judul" not in data or "abstrak" not in data:
        return jsonify({"error": "Harap kirim 'judul' dan 'abstrak'"}), 400

    judul = data["judul"]
    abstrak = data["abstrak"]

    try:
        kategori, confidence = predict(judul, abstrak)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "kategori": kategori,
        "confidence": confidence
    })

# Passenger WSGI (cPanel)
application = app

# Run local (port 5000)
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
