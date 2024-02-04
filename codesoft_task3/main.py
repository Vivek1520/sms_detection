import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm



data = pd.read_csv('spam.csv', encoding='latin-1')


data.drop_duplicates(inplace=True)


data['label'] = data['v1'].map({'ham': 'ham', 'spam': 'spam'})


X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['label'], test_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer()


X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)


classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)


X_test_tfidf = tfidf_vectorizer.transform(X_test)


y_pred = classifier.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)


report = classification_report(y_test, y_pred, target_names=['Legitimate SMS', 'Spam SMS'])


for i in tqdm(range(10, 101, 10), desc='Progress:', leave=True):
    pass


print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)