# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase with the correct URL
cred = credentials.Certificate("C:/Users/Me/Desktop/dpf/alumni/backend/alumni-connect-firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://alumni-connect-e6848-default-rtdb.firebaseio.com'
})

def prepare_user_data(user_data):
    """Prepare user data by combining relevant features"""
    features = []
    if user_data.get("skills"):
        features.append(str(user_data["skills"]))
    if user_data.get("language"):
        features.append(str(user_data["language"]))
    if user_data.get("location"):
        features.append(str(user_data["location"]))
    return " ".join(features)

def fetch_user_data():
    try:
        ref = db.reference('users')
        users = ref.get()
        
        if users is None:
            print("No users found in the database")
            return pd.DataFrame()
        
        data = []
        alumni_data = []
        
        for user_id, user_data in users.items():
            if user_data.get("userType") == "alumni":
                combined_features = prepare_user_data(user_data)
                alumni_data.append({
                    "user_id": user_id,
                    "features": combined_features,
                    "fullName": user_data.get("fullName", ""),
                    "email": user_data.get("email", ""),
                    "company": user_data.get("company", ""),
                    "location": user_data.get("location", "")  # Added location
                })
        
        print(f"Found {len(alumni_data)} alumni records")
        return pd.DataFrame(alumni_data)
    
    except Exception as e:
        print(f"Error fetching user data: {str(e)}")
        raise

def train_recommendation_model():
    # Fetch and prepare data
    df = fetch_user_data()
    
    if df.empty:
        print("No data available to train the model")
        return None, None, None
    
    # Vectorize the features
    vectorizer = CountVectorizer(stop_words='english')
    feature_matrix = vectorizer.fit_transform(df['features'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(feature_matrix)
    
    # Save the model, vectorizer, and DataFrame
    joblib.dump(cosine_sim, 'recommendation_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    df.to_pickle('alumni_data.pkl')
    
    print("Model training completed successfully!")
    return cosine_sim, vectorizer, df

def get_recommendations(current_user_data, cosine_sim, vectorizer, alumni_df, n_recommendations=3):
    """Get top N alumni recommendations for current user"""
    if cosine_sim is None or vectorizer is None or alumni_df is None or alumni_df.empty:
        return []
        
    # Prepare current user's features
    current_user_features = prepare_user_data(current_user_data)
    
    # Transform current user's features using the same vectorizer
    current_user_vector = vectorizer.transform([current_user_features])
    
    # Calculate similarity with all alumni
    similarities = cosine_similarity(current_user_vector, 
                                  vectorizer.transform(alumni_df['features']))
    
    # Get top N similar alumni
    similar_indices = similarities[0].argsort()[::-1][:n_recommendations]
    
    recommendations = []
    for idx in similar_indices:
        recommendations.append({
            'name': alumni_df.iloc[idx]['fullName'],
            'company': alumni_df.iloc[idx]['company'],
            'location': alumni_df.iloc[idx]['location'],
            'similarity_score': round(similarities[0][idx] * 100, 2)  # Convert to percentage
        })
    
    return recommendations

if __name__ == "__main__":
    try:
        # Train the model
        cosine_sim, vectorizer, alumni_df = train_recommendation_model()
        
        if cosine_sim is not None:
            # Example usage
            current_user = {
                "skills": "frontend",
                "language": "c++",
                "location": "US"
            }
            
            recommendations = get_recommendations(current_user, cosine_sim, vectorizer, alumni_df)
            print("\nTop 3 Alumni Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['name']}")
                print(f"   Company: {rec['company']}")
                print(f"   Location: {rec['location']}")
                print(f"   Match: {rec['similarity_score']}%")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

