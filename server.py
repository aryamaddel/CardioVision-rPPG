import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Allow cross-origin requests from the React Native app

@app.route('/analyze', methods=['POST'])
def analyze():
    print("Received a scan request!")

    # 1. Check if frames were uploaded
    if 'frames' not in request.files:
        return jsonify({"error": "No frames uploaded"}), 400

    frames = request.files.getlist('frames')
    frame_count = request.form.get('frame_count', 0)
    fps = request.form.get('fps', 10)

    print(f"Received {len(frames)} frames. Reported count: {frame_count}, FPS: {fps}")

    # 2. In a real scenario, you would process the images (faces) here
    # using rPPG algorithms (ICA, POS, etc.) to extract the pulse signal.
    # For now, we simulate processing time and return a mock accurate result.

    # 3. Simulated Response mapping
    # Returning high quality mock data so the app displays correctly
    response_data = {
        "bpm": 74.2,
        "rmssd": 45.8,
        "sdnn": 39.4,
        "lf_hf": 1.2, # Low stress (< 1.5)
        "confidence": 0.88,
        "ibi_array": [810, 815, 825, 805, 830, 810, 820, 800, 835, 815, 810, 825, 805, 830, 810, 820, 800, 835, 815],
        "stress_level": "low"
    }

    print("Analysis complete. Sending result...")
    return jsonify(response_data)

if __name__ == '__main__':
    # Run the server on all interfaces so the phone can connect
    print("Starting CardioVision rPPG Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
