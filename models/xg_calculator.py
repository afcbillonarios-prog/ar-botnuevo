import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

class xGCalculator:
    """
    A class to calculate Expected Goals (xG) for shots using machine learning.
    Features include shot location, body part, assist type, game situation, etc.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the xGCalculator.
        
        Args:
            model_path (str): Path to a pre-trained model file. If None, a new model is created.
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'x', 'y',                    # Shot coordinates (normalized 0-1)
            'distance_to_goal',          # Distance from goal in meters
            'angle_to_goal',             # Angle to goal in degrees
            'body_part',                 # 0: right foot, 1: left foot, 2: head, 3: other
            'assist_type',               # 0: none, 1: pass, 2: cross, 3: through ball, 4: rebound
            'game_situation',            # 0: open play, 1: set piece, 2: counter attack
            'defender_distance',         # Distance to nearest defender in meters
            'goalkeeper_position',       # 0: central, 1: left, 2: right, 3: off-line
            'shot_type',                 # 0: regular, 1: volley, 2: header, 3: penalty
            'is_first_time',             # 0: no, 1: yes (first-time shot)
            'is_header'                  # 0: no, 1: yes (redundant with body_part but sometimes used)
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = XGBRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
    
    def prepare_features(self, shot_data):
        """
        Prepare features from raw shot data.
        
        Args:
            shot_data (dict or pd.DataFrame): Raw shot data containing the required features.
            
        Returns:
            np.ndarray: Prepared features ready for model prediction.
        """
        if isinstance(shot_data, dict):
            # Convert single shot dict to DataFrame
            df = pd.DataFrame([shot_data])
        else:
            df = shot_data.copy()
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and order features
        X = df[self.feature_names].values
        
        # Scale features
        # Always fit the scaler if it hasn't been fitted yet, otherwise transform
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ == []:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the xG model.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets (0 or 1 for goal/no goal).
            X_val (np.ndarray): Validation features (optional).
            y_val (np.ndarray): Validation targets (optional).
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
    
    def predict_xg(self, shot_data):
        """
        Predict xG value for a shot.
        
        Args:
            shot_data (dict or pd.DataFrame): Shot data containing the required features.
            
        Returns:
            float: Predicted xG value (probability of goal, between 0 and 1).
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model or load a pre-trained model.")
        
        # Prepare features
        X = self.prepare_features(shot_data)
        
        # Get prediction
        xg_value = self.model.predict(X)[0]
        
        # Ensure value is between 0 and 1
        return max(0.0, min(1.0, xg_value))
    
    def predict_xg_batch(self, shots_data):
        """
        Predict xG values for multiple shots.
        
        Args:
            shots_data (pd.DataFrame): DataFrame containing shot data.
            
        Returns:
            np.ndarray: Array of predicted xG values.
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model or load a pre-trained model.")
        
        # Prepare features
        X = self.prepare_features(shots_data)
        
        # Get predictions
        xg_values = self.model.predict(X)
        
        # Ensure values are between 0 and 1
        return np.clip(xg_values, 0.0, 1.0)
    
    def calculate_total_xg(self, shots_data):
        """
        Calculate total xG for a team or player from multiple shots.
        
        Args:
            shots_data (pd.DataFrame): DataFrame containing shot data.
            
        Returns:
            float: Total xG value.
        """
        xg_values = self.predict_xg_batch(shots_data)
        return np.sum(xg_values)
    
    def save_model(self, path):
        """
        Save the trained model and scaler to disk.
        
        Args:
            path (str): Directory path where to save the model.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        joblib.dump(self.model, os.path.join(path, 'xg_model.joblib'))
        joblib.dump(self.scaler, os.path.join(path, 'xg_scaler.joblib'))
        print(f"xG model saved to {path}")
    
    def load_model(self, path):
        """
        Load a pre-trained model and scaler from disk.
        
        Args:
            path (str): Directory path or specific model file path.
        """
        model_path = os.path.join(path, 'xg_model.joblib') if os.path.isdir(path) else path
        scaler_path = os.path.join(path, 'xg_scaler.joblib') if os.path.isdir(path) else path.replace('model', 'scaler')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"xG model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model or scaler not found in {path}")

# Example usage
if __name__ == "__main__":
    # Example: Creating and training an xG model with synthetic data
    print("Creating synthetic training data for xG model...")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 2000
    
    # Features: shot characteristics
    X = np.zeros((n_samples, len(xGCalculator().feature_names)))
    
    # x, y coordinates (normalized 0-1 pitch)
    X[:, 0] = np.random.beta(2, 5, n_samples)  # x: more shots near opponent goal
    X[:, 1] = np.random.beta(2, 2, n_samples) * 0.8 + 0.1  # y: across width
    
    # Derived features
    X[:, 2] = np.sqrt((X[:, 0] - 1)**2 + (X[:, 1] - 0.5)**2) * 105  # distance_to_goal (m)
    X[:, 3] = np.degrees(np.arctan2(7.32, X[:, 2]))  # angle_to_goal (simplified)
    
    # Other features
    X[:, 4] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.4, 0.15, 0.05])  # body_part
    X[:, 5] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.15, 0.1, 0.05])  # assist_type
    X[:, 6] = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1])  # game_situation
    X[:, 7] = np.random.exponential(5, n_samples)  # defender_distance
    X[:, 8] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.2, 0.2, 0.1])  # goalkeeper_position
    X[:, 9] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.1, 0.15, 0.05])  # shot_type
    X[:, 10] = np.random.binomial(1, 0.3, n_samples)  # is_first_time
    X[:, 11] = (X[:, 4] == 2).astype(int)  # is_header (when body_part is head)
    
    # Create realistic xG values based on features
    # Closer shots, better angles, headers/volleys from crosses have higher xG
    xg_true = (
        0.1 * (1 - X[:, 0]) +  # Better position (higher x)
        0.1 * (1 - abs(X[:, 1] - 0.5) * 2) +  # Central position
        0.3 * np.exp(-X[:, 2] / 20) +  # Distance decay
        0.2 * (X[:, 3] / 30) +  # Angle factor
        0.1 * (X[:, 4] == 2).astype(float) +  # Header bonus
        0.1 * (X[:, 5] == 2).astype(float) +  # Cross assist bonus
        0.1 * (X[:, 9] == 1).astype(float) +  # Volley bonus
        np.random.normal(0, 0.05, n_samples)  # Noise
    )
    
    # Ensure xG is between 0 and 1
    xg_true = np.clip(xg_true, 0.01, 0.95)
    
    # Generate actual goals (binary) based on xG probability
    y = np.random.binomial(1, xg_true)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    xg_calc = xGCalculator()
    xg_calc.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    y_pred = xg_calc.model.predict(xg_calc.scaler.transform(X_test))
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"xG Model MAE: {mae:.4f}")
    print(f"xG Model RMSE: {rmse:.4f}")
    
    # Example prediction
    sample_shot = {
        'x': 0.85,           # Near penalty spot
        'y': 0.5,            # Central
        'distance_to_goal': 12.0,  # Meters
        'angle_to_goal': 15.0,     # Degrees
        'body_part': 0,      # Right foot
        'assist_type': 1,    # Pass
        'game_situation': 0, # Open play
        'defender_distance': 3.0,  # Meters
        'goalkeeper_position': 0,  # Central
        'shot_type': 0,      # Regular
        'is_first_time': 0,
        'is_header': 0
    }
    
    xg_value = xg_calc.predict_xg(sample_shot)
    
    print("\nSample shot xG prediction:")
    print(f"Shot location: ({sample_shot['x']:.2f}, {sample_shot['y']:.2f})")
    print(f"Distance to goal: {sample_shot['distance_to_goal']}m")
    print(f"Angle to goal: {sample_shot['angle_to_goal']:.1f}°")
    print(f"Predicted xG: {xg_value:.3f}")
    
    # Example batch prediction
    shots_batch = pd.DataFrame([
        sample_shot,
        {
            'x': 0.95, 'y': 0.5, 'distance_to_goal': 6.0, 'angle_to_goal': 30.0,
            'body_part': 2, 'assist_type': 2, 'game_situation': 0,
            'defender_distance': 1.0, 'goalkeeper_position': 1,
            'shot_type': 2, 'is_first_time': 1, 'is_header': 1
        }
    ])
    
    xg_batch = xg_calc.predict_xg_batch(shots_batch)
    total_xg = xg_calc.calculate_total_xg(shots_batch)
    
    print(f"\nBatch prediction:")
    print(f"Shot 1 xG: {xg_batch[0]:.3f}")
    print(f"Shot 2 xG: {xg_batch[1]:.3f}")
    print(f"Total xG: {total_xg:.3f}")
    
    # Save model
    xg_calc.save_model('./saved_model')