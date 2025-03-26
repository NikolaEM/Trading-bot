import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # For saving the trained model

def train_model(data_path):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    features = ['SMA_20', 'SMA_50', 'RSI']
    X = df[features]
    y = df['Signal']
    
    # Split the data (using chronological order; no shuffling for time series)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc:.2f}")
    
    # Save the model for later use
    joblib.dump(model, "model_rf_gold.pkl")
    print("Model saved as model_rf_gold.pkl")

if __name__ == "__main__":
    train_model("data/gold_data_processed.csv")