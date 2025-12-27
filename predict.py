import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Ask user for input
text = input("Enter email message: ")

# Predict
vec = vectorizer.transform([text])
prediction = model.predict(vec)[0]

if prediction == 1:
    print("Prediction: Spam")
else:
    print("Prediction: Not Spam")
