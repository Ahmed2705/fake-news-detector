import joblib
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, classification_report
nlp = spacy.load('en_core_web_sm')
def clean_text(training_data):
    stop_words = set(stopwords.words('english'))
    training_data['text'] = training_data['text'].replace(r'[^\w\s]', '', regex=True)
    training_data['text'] = training_data['text'].replace(r'(<br\s/?>|br\s/?|/br)', '', regex=True)
    training_data['text'] = training_data['text'].apply( lambda x: " ".join([word for word in x.split() if word not in stop_words]))
    return training_data


fake=pd.read_csv("Fake.csv")
true=pd.read_csv("True.csv")

fake = clean_text(fake)
true = clean_text(true)


fake['label'] = 0
true['label'] = 1

df = pd.concat([fake, true]).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)

df['content'] = df['title'] + " " + df['text']

stop_words = set(stopwords.words('english'))

def lemmatize_text(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    return " ".join(lemmas)

print("Preprocessing texts (Please Wait)...")
df['clean_text'] = df['content'].apply(lemmatize_text)

x_train,x_test,y_train,y_test = train_test_split(df['clean_text'],df['label'],test_size=0.2,random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(x_train)
X_test_tfidf = vectorizer.transform(x_test)
print("Training SVM Model...")
model=svm.LinearSVC()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print("\n📊 Model Evaluation Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("F1-score:", round(f1_score(y_test, y_pred)*100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')










