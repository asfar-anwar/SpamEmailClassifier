# Spam Email Classifier

This is a simple Machine Learning project that checks whether an email message is **Spam** or **Not Spam**.
The project uses Python and basic Machine Learning to learn from example emails and make predictions


## Project Structure
- `train.py`: Loads the dataset, cleans the data, and trains the model.
- `predict.py`: A script to test the trained model with user input.
- `spam.csv`: The dataset used for training.
- `requirements.txt`: List of Python libraries needed to run the project.
- `spam_model.pkl`: The saved machine learning model.
- `vectorizer.pkl`: The saved text vectorizer.


## Technologies Used
- `Python`
- `Pandas`  
- `Scikit-learn` 
- `NLP (CountVectorizer)` 


##  How to Run the Project

Step 1: Install requirements
    Open your terminal and run:
pip install -r requirements.txt

Step 2: Train the model  
    python train.py  

Step 3: Predict spam or not spam  
    python predict.py 


## Features
- Cleans the email data before training
- Uses Naive Bayes algorithm
- Predicts whether an email is spam or not
- Easy to understand for beginners


## Output Example

Enter email message: You have won a free prize  
Prediction: Spam  

Enter email message: Please send me the notes  
Prediction: Not Spam 


## Author
Asfar Anwar