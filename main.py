import pandas as pd
import string
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import numpy as np

# Define spam indicators
SPAM_INDICATORS = {
    'urgency': ['urgent', 'immediate', 'act now', 'limited time', 'expires soon', 'today only'],
    'money': ['cash', 'money', '$', '₹', 'price', 'cost', 'free', 'discount', 'offer', 'cheap'],
    'prizes': ['won', 'winner', 'congratulations', 'selected', 'reward', 'prize', 'gift'],
    'suspicious_terms': ['click here', 'verify account', 'bank details', 'loan', 'credit', 'subscription'],
    'pressure': ['don\'t miss', 'last chance', 'only for you', 'exclusive', 'special offer']
}

def enhanced_clean_text(text):
    # Convert to lowercase
    text = str(text).lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def calculate_spam_score(message):
    message = message.lower()
    score = 0
    total_indicators = 0
    
    # Check for spam indicators
    for category, indicators in SPAM_INDICATORS.items():
        for indicator in indicators:
            if indicator in message:
                score += 1
            total_indicators += 1
    
    # Additional checks
    if re.search(r'\d{10}', message):  # Phone numbers
        score += 2
    if re.search(r'https?://\S+|www\.\S+', message):  # URLs
        score += 2
    if message.count('!') > 2:  # Multiple exclamation marks
        score += 1
    if re.search(r'[A-Z]{3,}', message):  # CAPS LOCK
        score += 1
        
    # Normalize score between 0 and 1
    normalized_score = score / (total_indicators + 5)  # +5 for additional checks
    return normalized_score

def extract_features(text):
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
        'special_char_count': sum(1 for c in text if not c.isalnum() and not c.isspace()),
        'url_count': len(re.findall(r'https?://\S+|www\.\S+', text)),
        'email_count': len(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)),
        'phone_count': len(re.findall(r'\d{10}|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}', text)),
        'exclamation_count': text.count('!')
    }
    return pd.Series(features)

def load_and_train_model():
    try:
        # Load the dataset
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        
        # Convert labels to binary
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        # Clean the text data
        df['cleaned_text'] = df['text'].apply(enhanced_clean_text)
        
        # Extract additional features
        feature_df = df['text'].apply(extract_features)
        
        # Create TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2
        )
        text_features = tfidf.fit_transform(df['cleaned_text'])
        
        # Combine all features
        X = np.hstack([
            text_features.toarray(),
            feature_df.values
        ])
        y = df['label'].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print("\n📊 Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, tfidf, feature_df.columns
    
    except FileNotFoundError:
        print("❌ Error: spam.csv file not found!")
        raise
    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        raise

def predict_message(msg, model, vectorizer, feature_columns, threshold=0.5):
    # Clean the message
    cleaned_msg = enhanced_clean_text(msg)
    
    # Extract features
    features = extract_features(msg)
    
    # Get TF-IDF features
    text_features = vectorizer.transform([cleaned_msg]).toarray()
    
    # Combine features
    X = np.hstack([
        text_features,
        features.values.reshape(1, -1)
    ])
    
    # Get model prediction
    model_prob = model.predict_proba(X)[0][1]
    
    # Get rule-based spam score
    spam_score = calculate_spam_score(msg)
    
    # Combine scores (80% model, 20% rules)
    combined_score = (0.8 * model_prob) + (0.2 * spam_score)
    
    confidence = round(combined_score * 100, 2)
    
    if combined_score > threshold:
        return f"🔴 Spam (Confidence: {confidence}%)"
    else:
        return f"🟢 Ham (Confidence: {100-confidence}%)"

def main():
    print("\n🤖 Welcome to Enhanced Spam Detector")
    print("==================================")
    
    try:
        print("\n🔄 Loading and training the model...")
        model, vectorizer, feature_columns = load_and_train_model()
        
        print("\n✨ Model loaded successfully!")
        print("--------------------------------")
        
        while True:
            user_input = input("\n💬 Enter a message to check (or type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                print("\n👋 Thank you for using the Spam Detector!")
                break
            
            try:
                result = predict_message(user_input, model, vectorizer, feature_columns)
                print("\n🔎 Prediction:", result)
                
                # Show detailed analysis
                spam_indicators_found = []
                for category, indicators in SPAM_INDICATORS.items():
                    found = [ind for ind in indicators if ind in user_input.lower()]
                    if found:
                        spam_indicators_found.append(f"⚠️ {category.title()}: {', '.join(found)}")
                
                if spam_indicators_found:
                    print("\n📝 Analysis:")
                    for indicator in spam_indicators_found:
                        print(indicator)
                        
            except Exception as e:
                print(f"❌ Error processing message: {str(e)}")
                print("Please try again with a different message.")
                
    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
        print("Please ensure spam.csv file is present and try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")