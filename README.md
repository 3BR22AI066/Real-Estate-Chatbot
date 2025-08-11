# Real Estate Chatbot 🏠💬

A smart chatbot that helps users find properties based on their requirements, with natural language processing and machine learning capabilities.

![Screenshot 2025-04-16 185813](https://github.com/user-attachments/assets/e161bee6-e5c0-4597-9714-f126828adede)

![Screenshot 2025-04-16 185924](https://github.com/user-attachments/assets/90e24078-1797-4afa-972d-c0f760efa8a5)

## Features ✨

- **Natural Language Processing**: Understands queries like "3BHK in Mumbai under ₹50 lakhs"
- **Property Search**: Finds matching properties using ML-powered recommendations
- **General Knowledge**: Answers questions about real estate concepts and locations
- **Conversation History**: Maintains context of the conversation
- **Responsive Design**: Works on both desktop and mobile devices

## Tech Stack 🛠️

### Backend
- Python
- Flask (REST API)
- Scikit-learn (ML models)
- Pandas (Data processing)
- Joblib (Model persistence)

### Frontend
- React.js
- Axios (API calls)
- CSS3 (Styling)

### Machine Learning Model 🤖

#### Model Architecture
- **Algorithm**: K-Nearest Neighbors (KNN) Classifier
- **Purpose**: Property recommendation based on similarity
- **Input Features**:
  - Price (standardized)
  - Bedrooms
  - Bathrooms
  - House size (sqft)
  - City (label encoded)
  - State (label encoded)
#### Dataset-Source:-https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset
