from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "minimal server working"})

if __name__ == "__main__":
    print("Starting minimal test server...")
    app.run(debug=False, host="0.0.0.0", port=4790)
