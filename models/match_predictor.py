import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class MatchPredictor:
    """
    A class to predict match outcomes (win/draw/loss) using XGBoost.
    Features include team form, xG, xGA, possession, pressing, injuries, shots.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the MatchPredictor.
        
        Args:
            model_path (str): Path to a pre-trained model file. If None, a new model is created.
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'team_form_home', 'team_form_away',
            'xG_home', 'xG_away',
            'xGA_home', 'xGA_away',
            'possession_home', 'possession_away',
            'pressing_home', 'pressing_away',
            'injuries_home', 'injuries_away',
            'shots_home', 'shots_away',
            'shots_on_target_home', 'shots_on_target_away'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                objective='multi:softprob',
                num_class=3,  # 0: loss, 1: draw, 2: win (from home team perspective)
                random_state=42
            )
    
    def prepare_features(self, match_data):
        """
        Prepare features from raw match data.
        
        Args:
            match_data (dict or pd.DataFrame): Raw match data containing the required features.
            
        Returns:
            np.ndarray: Prepared features ready for model prediction.
        """
        if isinstance(match_data, dict):
            # Convert single match dict to DataFrame
            df = pd.DataFrame([match_data])
        else:
            df = match_data.copy()
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and order features
        X = df[self.feature_names].values
        
        # Scale features
        try:
            # Try to transform using existing fitted scaler
            X_scaled = self.scaler.transform(X)
        except:
            # If scaler is not fitted, fit it first
            X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the match predictor model.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels (0: loss, 1: draw, 2: win).
            X_val (np.ndarray): Validation features (optional).
            y_val (np.ndarray): Validation labels (optional).
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation set if provided
        eval_set = [(X_train_scaled, y_train)]
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set.append((X_val_scaled, y_val))
        
        # Train model
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=False
        )
    
    def predict(self, match_data):
        """
        Predict match outcome probabilities.
        
        Args:
            match_data (dict or pd.DataFrame): Match data containing the required features.
            
        Returns:
            dict: Probabilities for each outcome (loss, draw, win) from home team perspective.
        """
        # Check if model is fitted
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained. Please train the model or load a pre-trained model.")
        
        # Prepare features
        X = self.prepare_features(match_data)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)[0]
        
        # Return as dictionary
        return {
            'prob_loss': probabilities[0],
            'prob_draw': probabilities[1],
            'prob_win': probabilities[2]
        }
    
    def predict_outcome(self, match_data):
        """
        Predict the most likely match outcome.
        
        Args:
            match_data (dict or pd.DataFrame): Match data containing the required features.
            
        Returns:
            str: Predicted outcome ('loss', 'draw', or 'win') from home team perspective.
        """
        probs = self.predict(match_data)
        outcome_idx = np.argmax([probs['prob_loss'], probs['prob_draw'], probs['prob_win']])
        outcomes = ['loss', 'draw', 'win']
        return outcomes[outcome_idx]
    
    def save_model(self, path):
        """
        Save the trained model and scaler to disk.
        
        Args:
            path (str): Directory path where to save the model.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        joblib.dump(self.model, os.path.join(path, 'match_predictor_model.joblib'))
        joblib.dump(self.scaler, os.path.join(path, 'match_predictor_scaler.joblib'))
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a pre-trained model and scaler from disk.
        
        Args:
            path (str): Directory path or specific model file path.
        """
        model_path = os.path.join(path, 'match_predictor_model.joblib') if os.path.isdir(path) else path
        scaler_path = os.path.join(path, 'match_predictor_scaler.joblib') if os.path.isdir(path) else path.replace('model', 'scaler')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model or scaler not found in {path}")

# Example usage
if __name__ == "__main__":
    # Example: Creating and training a model with synthetic data
    print("Creating synthetic training data...")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Features: team form, xG, xGA, possession, pressing, injuries, shots
    X = np.random.rand(n_samples, len(MatchPredictor().feature_names))
    
    # Create realistic relationships for target variable
    # Home advantage: higher xG, possession, etc. increase win probability
    home_advantage = (
        X[:, 2] * 0.3 - X[:, 3] * 0.2 +  # xG_home - xG_away weight
        X[:, 6] * 0.2 - X[:, 7] * 0.15 + # possession_home - possession_away
        X[:, 8] * 0.15 - X[:, 9] * 0.1   # pressing_home - pressing_away
    )
    
    # Convert to probabilities and sample outcomes
    prob_win = 1 / (1 + np.exp(-(home_advantage + 0.5)))  # Sigmoid with home bias
    prob_draw = np.full(n_samples, 0.25)  # Base draw probability
    prob_loss = 1 - prob_win - prob_draw
    
    # Normalize probabilities
    total = prob_win + prob_draw + prob_loss
    prob_win /= total
    prob_draw /= total
    prob_loss /= total
    
    # Sample outcomes
    y = np.array([np.random.choice([0, 1, 2], p=[l, d, w]) 
                  for l, d, w in zip(prob_loss, prob_draw, prob_win)])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    predictor = MatchPredictor()
    predictor.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    y_pred = predictor.model.predict(predictor.scaler.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")
    
    # Example prediction
    sample_match = {
        'team_form_home': 0.8,
        'team_form_away': 0.6,
        'xG_home': 1.8,
        'xG_away': 1.2,
        'xGA_home': 0.9,
        'xGA_away': 1.1,
        'possession_home': 55,
        'possession_away': 45,
        'pressing_home': 0.7,
        'pressing_away': 0.5,
        'injuries_home': 2,
        'injuries_away': 3,
        'shots_home': 15,
        'shots_away': 10,
        'shots_on_target_home': 6,
        'shots_on_target_away': 3
    }
    
    probs = predictor.predict(sample_match)
    outcome = predictor.predict_outcome(sample_match)
    
    print("\nSample match prediction:")
    print(f"Win probability: {probs['prob_win']:.3f}")
    print(f"Draw probability: {probs['prob_draw']:.3f}")
    print(f"Loss probability: {probs['prob_loss']:.3f}")
    print(f"Predicted outcome: {outcome}")
    
    # Save model
    predictor.save_model('./saved_model')