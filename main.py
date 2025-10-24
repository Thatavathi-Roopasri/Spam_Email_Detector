import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Step 2: Keep only necessary columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 3: Add extra spam messages
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

# Duplicate new spam messages to boost importance
extra_data = pd.concat([extra_data] * 5, ignore_index=True)

# Add the extra data to the original dataset
df = pd.concat([df, extra_data], ignore_index=True)

# Step 4: Convert labels to binary (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 5: Text cleaning
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

# Step 6: Split data into training and test sets
X = df['cleaned_message']
y = df['label']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Vectorize text (fit only on training data)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# Step 8: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test)
print("\n✅ Model Evaluation:")
print("📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 10: Predict user input
def predict_message(msg):
    cleaned_msg = clean_text(msg)
    vectorized_msg = vectorizer.transform([cleaned_msg])
    prediction = model.predict(vectorized_msg)[0]
    return "🔴 Spam" if prediction == 1 else "🟢 Ham"

# Step 11: Loop for predictions
while True:
    user_input = input("\n💬 Enter a message to check (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("👋 Exiting the spam detector.")
        break
    result = predict_message(user_input)
    print("🔎 Prediction:", result)
