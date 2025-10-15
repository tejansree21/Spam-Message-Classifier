# Spam Message Classifier using Machine Learning

This project classifies SMS messages as **Spam** or **Ham (Not Spam)** using **Machine Learning**.  
It uses text preprocessing with **TF-IDF Vectorization** and a **Naive Bayes classifier** for efficient spam detection.

---

## Dataset Overview

- **Source:** [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Dataset Size:** ~5,500 SMS messages  
- **Classes:**
  - `ham` → Legitimate message  
  - `spam` → Unsolicited/advertisement message  

---

## Project Workflow

1. **Load Dataset** from URL (UCI Spam Collection)  
2. **Data Exploration**
   - Class balance visualization  
   - Text inspection  
3. **Data Preprocessing**
   - Label encoding (`ham` → 0, `spam` → 1)  
   - Train-test split  
4. **Feature Extraction**
   - TF-IDF Vectorization to convert text into numerical form  
5. **Model Training**
   - Naive Bayes (MultinomialNB)  
6. **Evaluation**
   - Accuracy, Confusion Matrix, and Classification Report  
7. **Testing**
   - Predict spam/ham for custom user input messages  
8. **Model Saving**
   - Save trained model and vectorizer with Joblib  

---

## Model Performance

| Metric | Score |
|:-------|:------:|
| **Accuracy** | ~0.97 |
| **Precision** | ~0.96 |
| **Recall** | ~0.95 |
| **F1-Score** | ~0.96 |

*(Values may slightly vary per run.)*

---

## Visualization

Confusion Matrix of the model predictions:

python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')

## Technologies Used
Python 
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
Joblib


##  How to Run

1. Clone the repository:
   git clone https://github.com/<your-username>/spam-message-classifier.git
cd spam-message-classifier

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Notebook:
   jupyter notebook Spam_Message_Classifier.ipynb

## Results
- Achieved 97% accuracy using Multinomial Naive Bayes
- Successfully classifies SMS as Spam or Ham
- Model ready for deployment in web or mobile spam filters
