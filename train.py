import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# 1. Load with encoding to avoid UnicodeDecodeError
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# --- DATA CLEANING SECTION ---
# 2. Drop any rows where text or label might be missing
df = df.dropna()

# 3. Remove duplicate entries (common in spam datasets)
df = df.drop_duplicates(keep='first')

# 4. Optional: Clean white spaces from text
df['text'] = df['text'].str.strip()
# -----------------------------

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Vectorize
cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

# Train model
model = MultinomialNB()
model.fit(x_train_cv, y_train)

# Test
pred = model.predict(x_test_cv)
print("Cleaned Data Accuracy:", accuracy_score(y_test, pred))

# Save model
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("Training completed successfully with cleaned data.")