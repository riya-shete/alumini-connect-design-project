from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from train_model import get_recommendations

app = Flask(__name__)

# Allow frontend requests from localhost:3000
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load the trained model and data
try:
    cosine_sim = joblib.load('recommendation_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    alumni_df = pd.read_pickle('alumni_data.pkl')
    print("✅ Model and data loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model files: {str(e)}")
    cosine_sim, vectorizer, alumni_df = None, None, None

# Ensure all responses have the necessary CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    return response

# Define a home route
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Alumni Recommendation API!"})

# Handle preflight OPTIONS requests explicitly
@app.route('/get_recommendations', methods=['OPTIONS', 'POST'])
def handle_recommendations():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'Preflight response'})
    
    elif request.method == 'POST':
        """API to get alumni recommendations based on user input."""
        try:
            current_user = request.json
            print(f"📩 Received user data: {current_user}")

            # Validate input data
            required_keys = {'skills', 'language', 'location'}
            if not required_keys.issubset(current_user):
                return jsonify({
                    'status': 'error',
                    'message': 'Missing required user data'
                }), 400

            if cosine_sim is None or vectorizer is None or alumni_df is None or alumni_df.empty:
                return jsonify({
                    'status': 'error',
                    'message': 'Model or data not loaded or is empty'
                }), 500

            # Get recommendations
            recommendations = get_recommendations(current_user, cosine_sim, vectorizer, alumni_df)
            print(f"✅ Recommendations generated: {recommendations}")

            return jsonify({
                'status': 'success',
                'recommendations': recommendations
            })

        except Exception as e:
            print(f"❌ Error processing request: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

if __name__ == '__main__':
    if cosine_sim is not None and vectorizer is not None and alumni_df is not None and not alumni_df.empty:
        app.run(host='0.0.0.0', port=5000, debug=True)  # Running on port 5000
    else:
        print("❌ Flask app cannot run without loading the model or data.") 