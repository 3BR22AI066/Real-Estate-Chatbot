from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime
import re
from sklearn.neighbors import NearestNeighbors
import warnings

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app)

# Configuration
warnings.filterwarnings("ignore")
MODEL_DIR = 'backend/models'
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths
MODEL_PATH = os.path.join(MODEL_DIR, 'real_estate_knn_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoders.joblib')
PROCESSED_DATA_PATH = os.path.join(MODEL_DIR, 'processed_realtor_data.csv')

# Global variables
reference_data = pd.DataFrame()
model = None
scaler = None
label_encoders = {'city': None, 'state': None}
location_nn = None
conversation_history = []

# Load data and models
def load_resources():
    global reference_data, model, scaler, label_encoders, location_nn
    
    try:
        # Load and clean data
        reference_data = pd.read_csv(PROCESSED_DATA_PATH)
        reference_data = reference_data[reference_data['price'] > 0].dropna()
        
        # Load ML artifacts
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoders = joblib.load(ENCODER_PATH)
        
        # Prepare location features
        location_features = reference_data[['city_encoded', 'state_encoded']].values
        location_nn = NearestNeighbors(n_neighbors=3).fit(location_features)
        
        print("Successfully loaded all resources")
    except Exception as e:
        print(f"Error loading resources: {str(e)}")

load_resources()

# Helper functions
def find_similar_location(city=None, state=None):
    try:
        if city:
            # Try exact match first
            if city.title() in reference_data['city'].unique():
                return city.title(), state
            
            # Find similar named cities
            similar_cities = [c for c in reference_data['city'].unique() 
                            if city.lower() in c.lower()]
            if similar_cities:
                return similar_cities[0], state
            
            # Find nearest encoded location
            if state and state.title() in reference_data['state'].unique():
                state_data = reference_data[reference_data['state'] == state.title()]
                avg_encoding = state_data[['city_encoded', 'state_encoded']].mean().values
            else:
                avg_encoding = reference_data[['city_encoded', 'state_encoded']].mean().values
            
            _, indices = location_nn.kneighbors([avg_encoding])
            nearest = reference_data.iloc[indices[0][0]]
            return nearest['city'], nearest['state']
        return city, state
    except:
        return None, None

def extract_search_parameters(message):
    params = {
        'bed': None,
        'bath': None,
        'price': None,
        'city': None,
        'state': None
    }
    
    message = message.lower()
    
    # Extract bedroom count
    bed_match = re.search(r'(\d+)\s*bed|bedroom', message)
    if bed_match:
        params['bed'] = int(bed_match.group(1))
    
    # Extract bathroom count
    bath_match = re.search(r'(\d+)\s*bath|bathroom', message)
    if bath_match:
        params['bath'] = int(bath_match.group(1))
    
    # Extract price
    price_match = re.search(r'(?:under|below|less than)\s*\$?(\d{1,3}(?:,\d{3})*)', message)
    if price_match:
        params['price'] = float(price_match.group(1).replace(',', ''))
    
    # Extract location
    location_match = re.search(r'in\s+([a-zA-Z\s]+(?:\s*,\s*[a-zA-Z\s]+)?)', message)
    if location_match:
        location = location_match.group(1).strip()
        parts = [part.strip().title() for part in location.split(',')]
        
        if len(parts) > 1:
            params['city'], params['state'] = parts[0], parts[1]
        else:
            params['city'] = parts[0]
    
    return {k: v for k, v in params.items() if v is not None}

def find_properties(params):
    try:
        # Set defaults
        defaults = {
            'price': reference_data['price'].median(),
            'bed': 3,
            'bath': 2,
            'house_size': reference_data['house_size'].median()
        }
        params = {**defaults, **params}
        
        # Handle location
        params['city'], params['state'] = find_similar_location(
            params.get('city'), 
            params.get('state')
        )
        
        # Filter properties
        filtered = reference_data.copy()
        
        # Apply filters with progressive relaxation
        for attempt in range(3):
            temp_filtered = reference_data.copy()
            
            # Bed filter
            if 'bed' in params:
                min_bed = max(1, params['bed'] - attempt)
                temp_filtered = temp_filtered[temp_filtered['bed'] >= min_bed]
            
            # Price filter
            if 'price' in params:
                max_price = params['price'] * (1 + 0.5 * attempt)
                temp_filtered = temp_filtered[temp_filtered['price'] <= max_price]
            
            # Location filters
            if params.get('city'):
                try:
                    city_encoded = label_encoders['city'].transform([params['city']])[0]
                    temp_filtered = temp_filtered[temp_filtered['city_encoded'] == city_encoded]
                except:
                    pass
            
            if params.get('state'):
                try:
                    state_encoded = label_encoders['state'].transform([params['state']])[0]
                    temp_filtered = temp_filtered[temp_filtered['state_encoded'] == state_encoded]
                except:
                    pass
            
            if len(temp_filtered) > 0:
                filtered = temp_filtered
                break
        
        # If no matches, return affordable properties
        if len(filtered) == 0:
            filtered = reference_data.sort_values('price').head(5)
            return {
                'status': 'partial',
                'properties': filtered.to_dict(orient='records'),
                'message': 'No exact matches found. Showing affordable options.'
            }
        
        # Prepare features
        features = filtered[['price', 'bed', 'bath', 'house_size', 'city_encoded', 'state_encoded']]
        scaled_features = scaler.transform(features)
        
        # Create input vector
        input_vec = [
            params.get('price', defaults['price']),
            params.get('bed', defaults['bed']),
            params.get('bath', defaults['bath']),
            defaults['house_size'],
            label_encoders['city'].transform([params.get('city', 'Unknown')])[0] if params.get('city') else 0,
            label_encoders['state'].transform([params.get('state', 'Unknown')])[0] if params.get('state') else 0
        ]
        
        scaled_input = scaler.transform([input_vec])
        
        # Find similar properties
        nn = NearestNeighbors(n_neighbors=min(5, len(filtered)))
        nn.fit(scaled_features)
        distances, indices = nn.kneighbors(scaled_input)
        
        results = filtered.iloc[indices[0]].to_dict(orient='records')
        
        return {
            'status': 'success',
            'properties': results,
            'distances': distances[0].tolist()
        }
    except Exception as e:
        print(f"Search error: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }

# API Endpoints
@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "Real Estate Chatbot API",
        "endpoints": ["POST /chat"]
    })

@app.route('/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "status": "error",
                "message": "Please provide a 'message' in your request"
            }), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                "status": "error",
                "message": "Message cannot be empty"
            }), 400
        
        # Handle greetings
        if user_message.lower() in ['hi', 'hello', 'hey']:
            return jsonify({
                "status": "success",
                "response": "Hello! I can help you find properties. Try asking 'Find me a 3 bedroom house in San Francisco under $4000'",
                "data": None
            })
        
        # Extract parameters
        params = extract_search_parameters(user_message)
        
        # Find properties
        result = find_properties(params)
        
        # Format response
        if result['status'] == 'error':
            return jsonify({
                "status": "error",
                "message": result['message']
            }), 400
        
        properties = result.get('properties', [])[:3]
        if not properties:
            response = "No properties found matching your criteria."
        else:
            response = result.get('message', 'Here are some properties:') + "\n"
            for prop in properties:
                try:
                    bed = int(prop['bed'])
                    bath = int(prop['bath'])
                    price = f"${prop['price']:,.0f}" if prop['price'] > 0 else "Price not available"
                    size = f"{int(prop['house_size']):,} sqft" if not np.isnan(prop['house_size']) else "Size not available"
                    location = f"{prop.get('city', 'Unknown')}, {prop.get('state', '')}".strip(', ')
                    
                    response += f"\n- {bed} bed, {bath} bath, {size} in {location} for {price}"
                except Exception as e:
                    print(f"Error formatting property: {str(e)}")
                    continue
        
        # Store conversation
        conversation_history.append({
            'user': user_message,
            'bot': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            "status": "success",
            "response": response,
            "data": {
                "properties": properties,
                "status": result['status']
            }
        })
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)