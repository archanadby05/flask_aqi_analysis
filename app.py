from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load and preprocess the dataset
def load_and_clean_data():
    data = pd.read_csv('AQI_Data.csv')
    
    # Identify numeric columns only
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Handle missing values in numeric columns, fill with median
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    # Encode categorical AQI_Bucket to numerical labels
    label_encoder = LabelEncoder()
    if 'AQI_Bucket' in data.columns:
        data['AQI_Bucket'] = label_encoder.fit_transform(data['AQI_Bucket'])
    
    # Separate features and labels
    X = data.drop(['City', 'Date', 'AQI_Bucket'], axis=1, errors='ignore')
    y = data['AQI_Bucket']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'K-Means (Clustering)': KMeans(n_clusters=len(y_train.unique()))
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # If clustering, assign labels and evaluate using cluster labels as baseline
        if name == 'K-Means (Clustering)':
            y_pred = model.predict(X_test)
            y_pred = pd.Series(y_pred).map(lambda x: 1 if x in y_test.unique() else 0)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred, average='weighted')
        })
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    X_train, X_test, y_train, y_test = load_and_clean_data()
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
