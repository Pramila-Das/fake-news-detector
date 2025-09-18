import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("data.csv")

# Combine Headline + Body into single text field
df['content'] = df['Headline'].fillna('') + " " + df['Body'].fillna('')

# X = features, y = labels
X = df['content']
y = df['Label']   # 0 = Fake, 1 = Real

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Build pipeline (TF-IDF + Logistic Regression)
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# 4. Train
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("🔹 Model Performance:\n")
print(classification_report(y_test, y_pred))

# 6. Save the model
import pickle

# Save the entire pipeline safely
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

