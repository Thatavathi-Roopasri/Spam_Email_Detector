import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Add extra spam messages
extra_data = pd.DataFrame({
    'label': ['spam'] * 10,
    'message': [
        "Congratulations! You've won a free vacation to Dubai. Call now to claim!",
        "Urgent! Your account has been compromised. Click here to secure it immediately.",
        "Winner! You’ve been chosen for a $1000 Walmart gift card. Act fast!",
        "You have pending rewards. Click this link to access your prize now.",
        "Get a brand new iPhone for just ₹1! Limited time offer.",
        "This is not a scam! You've won $5,000. Contact us today.",
        "Win cash now! Reply 'WIN' to this message to claim ₹10,000.",
        "Click here to claim your free Netflix subscription. Limited time only!",
        "Lowest loan rates in the market! Apply now with no documents needed.",
        "You’ve been shortlisted for a free cruise. Confirm your ticket now!"
    ]
})
extra_data = pd.concat([extra_data] * 5, ignore_index=True)
df = pd.concat([df, extra_data], ignore_index=True)

# Label encoding
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

# Split and vectorize
X = df['cleaned_message']
y = df['label']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
model = MultinomialNB()
model.fit(X_train, y_train)

# Streamlit UI
st.title("📩 Spam Message Detector")
st.write("Enter a message below to check if it's Spam or Ham:")

user_input = st.text_area("💬 Message", height=100)

if st.button("🔍 Check"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    if prediction == 1:
        st.error("🔴 This is **SPAM**!")
    else:
        st.success("🟢 This is **HAM** (not spam).")
