import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Paths
RAW_DATA_PATH = 'backend/data/realtor-data.zip.csv'
PROCESSED_DATA_PATH = 'backend/models/processed_realtor_data.csv'
MODEL_DIR = 'backend/models'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'real_estate_knn_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoders.joblib')

# Load raw data
df = pd.read_csv(RAW_DATA_PATH)

# Keep necessary columns
df = df[['price', 'bed', 'bath', 'house_size', 'city', 'state']]
df.dropna(inplace=True)

# Label encode
le_city = LabelEncoder()
le_state = LabelEncoder()

df['city_encoded'] = le_city.fit_transform(df['city'])
df['state_encoded'] = le_state.fit_transform(df['state'])

# Save label encoders
joblib.dump({'city': le_city, 'state': le_state}, ENCODER_PATH)

# Save processed data for API reference
df.to_csv(PROCESSED_DATA_PATH, index=False)

# Feature selection and scaling
features = df[['price', 'bed', 'bath', 'house_size', 'city_encoded', 'state_encoded']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Save scaler
joblib.dump(scaler, SCALER_PATH)

# Model training
model = KNeighborsClassifier(n_neighbors=5)
model.fit(scaled_features, np.arange(len(df)))  # pseudo labels

# Save model
joblib.dump(model, MODEL_PATH)

print("✅ Model, scaler, and label encoders saved.")
