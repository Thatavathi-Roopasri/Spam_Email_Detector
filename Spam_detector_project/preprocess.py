import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Step 2: Keep only the important columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 3: Convert labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Clean the text (lowercase + remove punctuation)
# All messages are converted to lowercase (so “FREE” and “free” are treated the same).
# Punctuation (like ! , ? .) is removed, since it doesn’t add meaning for spam detection.
#Example: "Win $1000 NOW!!!" → "win 1000 now"
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

# Step 5: Split into training and test sets
# We divide the data into two parts:
# Training data (80%) → used to teach the model.
# Test data (20%) → used to check if the model learned correctly.
#This helps avoid cheating, because the model must predict on data it hasn’t seen before
X = df['cleaned_message']
y = df['label']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Convert text into numbers using TF-IDF
# Machines can’t read words directly, so we convert text into numbers.
# TF-IDF (Term Frequency – Inverse Document Frequency) gives importance to words:
# Common words like “the” get low weight.
# Unique or spammy words like “win, free, offer” get high weight.
# After this, each message becomes a vector (a list of numbers) that the machine can use.
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

print("✅ Preprocessing Completed")
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
