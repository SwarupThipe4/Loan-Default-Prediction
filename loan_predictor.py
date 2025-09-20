import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

class LoanPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def load_data(self, filepath):
        """Load and preprocess loan dataset"""
        df = pd.read_csv(filepath)
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Feature engineering
        df['debt_to_income'] = df['debt'] / np.maximum(df['income'], 1)
        df['credit_utilization'] = df['credit_used'] / np.maximum(df['credit_limit'], 1)
        
        # Select features
        features = ['income', 'debt', 'credit_score', 'employment_years', 
                   'debt_to_income', 'credit_utilization']
        
        X = df[features]
        y = df['default']
        
        return X, y
    
    def train(self, X, y):
        """Train the model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
        print(classification_report(y_test, y_pred))
    
    def predict(self, applicant_data):
        """Predict default probability"""
        df = pd.DataFrame([applicant_data])
        df['debt_to_income'] = df['debt'] / df['income']
        df['credit_utilization'] = df['credit_used'] / df['credit_limit']
        
        features = ['income', 'debt', 'credit_score', 'employment_years', 
                   'debt_to_income', 'credit_utilization']
        
        X = self.scaler.transform(df[features])
        probability = self.model.predict_proba(X)[0, 1]
        
        risk = 'HIGH' if probability > 0.6 else 'MEDIUM' if probability > 0.3 else 'LOW'
        
        return {
            'probability': round(probability, 3),
            'risk': risk,
            'decision': 'REJECT' if risk == 'HIGH' else 'REVIEW' if risk == 'MEDIUM' else 'APPROVE'
        }
    
    def save(self, filepath='model.pkl'):
        joblib.dump({'scaler': self.scaler, 'model': self.model}, filepath)
    
    def load(self, filepath='model.pkl'):
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.model = data['model']

# Create sample dataset
def create_dataset():
    np.random.seed(42)
    n = 5000
    
    data = {
        'income': np.random.lognormal(10.8, 0.6, n),
        'debt': np.random.lognormal(9.5, 0.8, n),
        'credit_score': np.clip(np.random.normal(650, 100, n), 300, 850),
        'employment_years': np.clip(np.random.exponential(4, n), 0, 40),
        'credit_limit': np.random.lognormal(9.2, 0.7, n),
        'credit_used': np.random.lognormal(8.5, 0.9, n)
    }
    
    df = pd.DataFrame(data)
    df['credit_used'] = np.minimum(df['credit_used'], df['credit_limit'])
    
    # Create target
    risk = ((df['debt'] / df['income']) * 0.4 + 
            ((700 - df['credit_score']) / 100) * 0.3 + 
            (df['credit_used'] / df['credit_limit']) * 0.3)
    
    df['default'] = (risk > np.percentile(risk, 80)).astype(int)
    
    # Optimize data types and precision to reduce file size
    df['income'] = df['income'].round(0).astype('int32')
    df['debt'] = df['debt'].round(0).astype('int32')
    df['credit_score'] = df['credit_score'].round(0).astype('int16')
    df['employment_years'] = df['employment_years'].round(1).astype('float32')
    df['credit_limit'] = df['credit_limit'].round(0).astype('int32')
    df['credit_used'] = df['credit_used'].round(0).astype('int32')
    df['default'] = df['default'].astype('int8')
    
    df.to_csv('loan_data.csv', index=False)
    return df

if __name__ == "__main__":
    # Create sample data if not exists
    try:
        df = pd.read_csv('loan_data.csv')
    except FileNotFoundError:
        df = create_dataset()
    
    # Train model
    predictor = LoanPredictor()
    X, y = predictor.load_data('loan_data.csv')
    predictor.train(X, y)
    predictor.save()
    
    # Test prediction
    test_applicant = {
        'income': 60000,
        'debt': 15000,
        'credit_score': 720,
        'employment_years': 3,
        'credit_limit': 8000,
        'credit_used': 2000
    }
    
    result = predictor.predict(test_applicant)
    print(f"\nTest Result: {result}")