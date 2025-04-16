import spacy
import re
import pandas as pd

nlp = spacy.load("en_core_web_sm")

# Load city list from dataset to improve accuracy
cities = pd.read_csv('C:/Users/katta/OneDrive/Desktop/real_estate_chatbot/backend/data/realtor-data.zip.csv')['city'].dropna().unique()
cities_lower = [city.lower() for city in cities]

def extract_preferences(message):
    doc = nlp(message)
    preferences = {
        "location": None,
        "min_price": None,
        "max_price": None,
        "beds": None
    }

    # Extract city (GPE = geopolitical entity)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            preferences["location"] = ent.text.lower()

    # If spaCy didn't catch a city, look for it manually
    if not preferences["location"]:
        for token in doc:
            if token.text.lower() in cities_lower:
                preferences["location"] = token.text.lower()
                break

    # Extract price
    prices = re.findall(r'\$?\d[\d,]*', message)
    prices = [int(p.replace('$', '').replace(',', '')) for p in prices]

    if prices:
        preferences["max_price"] = max(prices)

    # Extract bedrooms
    bed_match = re.search(r'(\d+)[ -]?(?:bed|bedroom)', message, re.IGNORECASE)
    if bed_match:
        preferences["beds"] = int(bed_match.group(1))

    return preferences
