# Loan Default Prediction

A machine learning system to predict loan default risk using financial data.

## Features
- **Predictive Model**: Random Forest classifier for loan default assessment
- **Feature Engineering**: Debt-to-income ratio and credit utilization features
- **Performance Metrics**: ROC-AUC scores and classification reports
- **Model Persistence**: Save/load trained models
- **Risk Assessment**: Categorizes risk as LOW/MEDIUM/HIGH with recommendations

## Tech Stack
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib

## Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/loan-default-prediction.git
cd loan-default-prediction
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the predictor**:
```bash
python loan_predictor.py
```

This will:
- Generate sample loan data
- Train the Random Forest model
- Evaluate model performance
- Save the trained model
- Test with a sample applicant

## Usage

```python
from loan_predictor import LoanPredictor

# Initialize and load trained model
predictor = LoanPredictor()
predictor.load('model.pkl')

# Assess loan applicant
applicant = {
    'income': 60000,
    'debt': 15000,
    'credit_score': 720,
    'employment_years': 3,
    'credit_limit': 8000,
    'credit_used': 2000
}

result = predictor.predict(applicant)
print(result)
# Output: {'probability': 0.0, 'risk': 'LOW', 'decision': 'APPROVE'}
```

## Dataset Format

The model expects CSV data with these columns:
- `income`: Annual income
- `debt`: Total debt amount
- `credit_score`: Credit score (300-850)
- `employment_years`: Years of employment
- `credit_limit`: Total credit limit
- `credit_used`: Amount of credit used
- `default`: Target variable (1=default, 0=no default)

## Model Performance

The trained model achieves:
- **ROC-AUC**: ~0.91
- **Accuracy**: ~89%
- **Precision**: 91% (non-default), 77% (default)
- **Recall**: 96% (non-default), 63% (default)

## Project Structure

```
loan-default-prediction/
├── loan_predictor.py    # Main prediction model
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── .gitignore          # Git ignore rules
├── loan_data.csv       # Sample dataset (generated)
└── model.pkl           # Trained model (generated)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License