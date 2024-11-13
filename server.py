import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
from flask_cors import CORS
import io  
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

csv_files = glob.glob("*.csv")
data = pd.read_csv(csv_files[0])

if csv_files:
    
    data = pd.read_csv(csv_files[0])
    print(f"loaded file:{csv_files[0]}")
else:
    print("No csv files found in directory")

    print(data.head())


    print(data.isnull().sum())


    data.columns = [col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') for col in data.columns]

    data.drop('customerID', axis=1, inplace=True)

    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    print(data.isnull().sum())

    data.dropna(inplace=True)

    sns.countplot(x='Churn', data=data)
    plt.title('Churn Distribution')
    plt.show()

    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[numerical_features].hist(figsize=(10, 7))
    plt.show()

    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Churn')

    le = LabelEncoder()

    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])  

    data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    print(data.head())

    x = data.drop('Churn', axis=1)
    y = data['Churn']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    y_pred_probs = rf.predict_proba(x_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    roc_auc = roc_auc_score(y_test, y_pred_probs)
    print(f"ROC-AUC score: {roc_auc:.2f}")

    importance = rf.feature_importances_
    features = x.columns
    feature_importance = pd.Series(importance, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.title('Feature Importance')
    plt.show()


    app = Flask(__name__)
    CORS(app)  

    @app.route('/process_csv', methods=['POST'])  
    def process_csv():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and file.filename.endswith('.csv'):
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            csv_data = pd.read_csv(stream)
            
            result = {
                'rows': len(csv_data),
                'columns': list(csv_data.columns),
                'preview': csv_data.head().to_dict('records')
            }
            
            return jsonify(result)
        
        return jsonify({'error': 'Invalid file type'}), 400

    if __name__ == '__main__':
        app.run(debug=True)