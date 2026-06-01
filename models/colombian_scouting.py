import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class ColombianScoutingSystem:
    """
    A comprehensive scouting system for Colombian football talent identification.
    Features:
    - Player clustering by playing style/position
    - Similarity-based player comparison
    - Potential prediction for European leagues
    - Undervalued player detection
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the Colombian Scouting System.
        
        Args:
            model_path (str): Path to pre-trained models. If None, new models are created.
        """
        self.scaler = StandardScaler()
        self.kmeans = None
        self.potential_model = None
        self.feature_names = [
            'edad', 'minutos', 'goles', 'xG', 'asistencias', 
            'pases', 'duelos', 'velocidad', 'xA', 'progresividad',
            'recuperaciones', 'intercepciones', 'faltas_recibidas'
        ]
        self.position_clusters = {
            0: 'Delantero Finalizador',
            1: 'Extremo Desequilibrante', 
            2: 'Mediocentro Creativo',
            3: 'Pivote Defensivo',
            4: 'Central Constructor',
            5: 'Lateral Ofensivo',
            6: 'Media Punta',
            7: 'Segundo Delantero'
        }
        
        if model_path and os.path.exists(model_path):
            self.load_models(model_path)
        else:
            # Initialize models
            self.kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
            self.potential_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )
    
    def prepare_features(self, player_data):
        """
        Prepare features from raw player data.
        
        Args:
            player_data (dict or pd.DataFrame): Raw player data.
            
        Returns:
            np.ndarray: Prepared features ready for modeling.
        """
        if isinstance(player_data, dict):
            df = pd.DataFrame([player_data])
        else:
            df = player_data.copy()
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            # Fill missing features with zeros or means for demonstration
            for feature in missing_features:
                df[feature] = 0
        
        # Select and order features
        X = df[self.feature_names].fillna(0).values
        
        # Scale features
        if self.kmeans is None or not hasattr(self.kmeans, 'cluster_centers_'):
            # If not fitted yet, fit the scaler
            X_scaled = self.scaler.fit_transform(X)
        else:
            # If already fitted, use the fitted scaler
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def train_clustering(self, player_data):
        """
        Train the player clustering model.
        
        Args:
            player_data (pd.DataFrame): DataFrame containing player statistics.
        """
        print("Training player clustering model...")
        
        # Prepare features
        X = self.prepare_features(player_data)
        
        # Fit scaler and KMeans
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        
        # Assign cluster labels
        player_data['cluster'] = self.kmeans.labels_
        player_data['position_role'] = player_data['cluster'].map(self.position_clusters)
        
        print(f"Clustering completed. Found {len(self.position_clusters)} player roles.")
        return player_data
    
    def predict_player_role(self, player_data):
        """
        Predict the playing role/position of a player.
        
        Args:
            player_data (dict or pd.DataFrame): Player statistics.
            
        Returns:
            str: Predicted player role.
        """
        if self.kmeans is None or not hasattr(self.kmeans, 'cluster_centers_'):
            raise ValueError("Clustering model not trained. Please train the model first.")
        
        # Prepare features
        X = self.prepare_features(player_data)
        
        # Predict cluster
        cluster = self.kmeans.predict(X)[0]
        
        # Return role
        return self.position_clusters.get(cluster, 'Unknown Role')
    
    def find_similar_players(self, target_player, player_database, top_n=10):
        """
        Find similar players based on playing style and statistics.
        
        Args:
            target_player (dict): Target player statistics.
            player_database (pd.DataFrame): Database of players to compare against.
            top_n (int): Number of similar players to return.
            
        Returns:
            pd.DataFrame: Top N similar players with similarity scores.
        """
        if self.kmeans is None or not hasattr(self.kmeans, 'cluster_centers_'):
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare features for target player
        target_features = self.prepare_features(target_player)
        
        # Prepare features for database
        db_features = self.prepare_features(player_database)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(target_features, db_features).flatten()
        
        # Get top N similar players (excluding the target if present in database)
        similar_indices = np.argsort(similarities)[::-1][:top_n+1]
        
        # Create results dataframe
        similar_players = player_database.iloc[similar_indices].copy()
        similar_players['similarity_score'] = similarities[similar_indices]
        
        # Remove target player if it's in the database (similarity score of 1.0)
        similar_players = similar_players[similar_players['similarity_score'] < 0.999]
        
        return similar_players.head(top_n)
    
    def train_potential_model(self, player_data, european_success_label):
        """
        Train a model to predict potential for success in European leagues.
        
        Args:
            player_data (pd.DataFrame): Player statistics.
            european_success_label (np.array): Binary labels (1: succeeded in Europe, 0: did not).
        """
        print("Training European potential prediction model...")
        
        # Prepare features
        X = self.prepare_features(player_data)
        
        # Train model
        self.potential_model.fit(X, european_success_label)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.potential_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 5 features for European potential:")
        print(feature_importance.head())
        
        return feature_importance
    
    def predict_european_potential(self, player_data):
        """
        Predict the likelihood of a player succeeding in European leagues.
        
        Args:
            player_data (dict or pd.DataFrame): Player statistics.
            
        Returns:
            float: Probability of success in European leagues (0-1).
        """
        if self.potential_model is None:
            raise ValueError("Potential model not trained. Please train the model first.")
        
        # Prepare features
        X = self.prepare_features(player_data)
        
        # Get probability
        probability = self.potential_model.predict_proba(X)[0][1]  # Probability of class 1 (success)
        
        return probability
    
    def detect_undervalued_players(self, player_data, market_values=None):
        """
        Detect potentially undervalued players based on performance vs. market value.
        
        Args:
            player_data (pd.DataFrame): Player statistics and current market values.
            market_values (pd.Series): Market values for players (optional).
            
        Returns:
            pd.DataFrame: Players sorted by performance-to-value ratio.
        """
        # If no market values provided, create a simple performance score
        if market_values is None:
            # Create a performance score based on key metrics
            performance_score = (
                player_data['goles'] * 0.3 +
                player_data['asistencias'] * 0.25 +
                player_data['xG'] * 0.2 +
                player_data['xA'] * 0.15 +
                player_data['duelos'] * 0.1  # Defensive contribution
            )
            
            # Normalize minutes played (more minutes = more reliable data)
            minutes_factor = np.minimum(player_data['minutos'] / 1500, 1.0)  # Cap at ~16 full games
            
            # Adjusted performance score
            player_data['adjusted_performance'] = performance_score * minutes_factor
            
            # Use age as inverse proxy for market value (younger = potentially higher value)
            # In reality, we'd use actual transfer market data
            value_proxy = 1 / (player_data['edad'] * 0.1 + 0.5)  # Younger players get higher value proxy
            
            # Calculate performance-to-value ratio
            player_data['value_ratio'] = player_data['adjusted_performance'] * value_proxy
            
        else:
            # If we have actual market values
            performance_score = (
                player_data['goles'] * 0.3 +
                player_data['asistencias'] * 0.25 +
                player_data['xG'] * 0.2 +
                player_data['xA'] * 0.15 +
                player_data['duelos'] * 0.1
            )
            player_data['value_ratio'] = performance_score / market_values
        
        # Sort by value ratio (higher = better performance for cost)
        return player_data.sort_values('value_ratio', ascending=False)
    
    def save_models(self, path):
        """
        Save trained models to disk.
        
        Args:
            path (str): Directory path where to save the models.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        joblib.dump(self.kmeans, os.path.join(path, 'kmeans_model.joblib'))
        joblib.dump(self.scaler, os.path.join(path, 'feature_scaler.joblib'))
        joblib.dump(self.potential_model, os.path.join(path, 'potential_model.joblib'))
        print(f"Colombian scouting models saved to {path}")
    
    def load_models(self, path):
        """
        Load pre-trained models from disk.
        
        Args:
            path (str): Directory path or specific model file path.
        """
        kmeans_path = os.path.join(path, 'kmeans_model.joblib') if os.path.isdir(path) else path
        scaler_path = os.path.join(path, 'feature_scaler.joblib') if os.path.isdir(path) else path.replace('kmeans_model', 'feature_scaler')
        potential_path = os.path.join(path, 'potential_model.joblib') if os.path.isdir(path) else path.replace('kmeans_model', 'potential_model')
        
        if os.path.exists(kmeans_path) and os.path.exists(scaler_path) and os.path.exists(potential_path):
            self.kmeans = joblib.load(kmeans_path)
            self.scaler = joblib.load(scaler_path)
            self.potential_model = joblib.load(potential_path)
            print(f"Colombian scouting models loaded from {path}")
        else:
            raise FileNotFoundError(f"One or more models not found in {path}")

# Example usage and demonstration
if __name__ == "__main__":
    print("="*60)
    print("COLOMBIAN FOOTBALL TALENT SCOUTING SYSTEM")
    print("="*60)
    
    # Create synthetic data representing Colombian Primera A players
    print("\nGenerating synthetic player data for demonstration...")
    np.random.seed(42)
    n_players = 200
    
    # Generate player data
    players_data = {
        'edad': np.random.randint(16, 35, n_players),
        'minutos': np.random.randint(0, 3500, n_players),
        'goles': np.random.poisson(8, n_players),
        'xG': np.random.gamma(2, 4, n_players),  # xG tends to be similar to goals but with variation
        'asistencias': np.random.poisson(5, n_players),
        'pases': np.random.randint(500, 3000, n_players),
        'duelos': np.random.randint(50, 400, n_players),
        'velocidad': np.random.normal(28, 4, n_players),  # km/h
        'xA': np.random.gamma(1.5, 3, n_players),
        'progresividad': np.random.randint(20, 200, n_players),  # progressive passes
        'recuperaciones': np.random.randint(10, 150, n_players),
        'intercepciones': np.random.randint(5, 100, n_players),
        'faltas_recibidas': np.random.randint(10, 80, n_players)
    }
    
    df_players = pd.DataFrame(players_data)
    
    # Ensure realistic ranges
    df_players['edad'] = np.clip(df_players['edad'], 16, 34)
    df_players['minutos'] = np.clip(df_players['minutos'], 0, 3800)
    df_players['goles'] = np.clip(df_players['goles'], 0, 25)
    df_players['xG'] = np.clip(df_players['xG'], 0, 20)
    df_players['asistencias'] = np.clip(df_players['asistencias'], 0, 20)
    df_players['velocidad'] = np.clip(df_players['velocidad'], 20, 38)
    
    print(f"Generated data for {len(df_players)} Colombian players")
    print(f"Age range: {df_players['edad'].min()}-{df_players['edad'].max()} years")
    print(f"Minutes played range: {df_players['minutos'].min()}-{df_players['minutos'].max()}")
    
    # Initialize the scouting system
    scouting_system = ColombianScoutingSystem()
    
    # Train clustering model
    print("\n" + "-"*50)
    print("TRAINING PLAYER CLUSTERING MODEL")
    print("-"*50)
    df_players_with_roles = scouting_system.train_clustering(df_players.copy())
    
    # Show cluster distribution
    print("\nPlayer Role Distribution:")
    role_counts = df_players_with_roles['position_role'].value_counts()
    for role, count in role_counts.items():
        print(f"  {role}: {count} players ({count/len(df_players_with_roles)*100:.1f}%)")
    
    # Example: Analyze a specific young player
    print("\n" + "-"*50)
    print("EXAMPLE PLAYER ANALYSIS")
    print("-"*50)
    
    # Create a sample young talented player (like a potential James Rodríguez or Luis Díaz)
    young_talent = {
        'edad': 19,
        'minutos': 1800,  # About half a season
        'goles': 12,
        'xG': 10.5,
        'asistencias': 8,
        'pases': 1200,
        'duelos': 180,
        'velocidad': 32.5,  # Very fast
        'xA': 7.2,
        'progresividad': 85,
        'recuperaciones': 45,
        'intercepciones': 25,
        'faltas_recibidas': 35
    }
    
    print("Sample Young Talent Profile:")
    for key, value in young_talent.items():
        print(f"  {key}: {value}")
    
    # Predict player role
    predicted_role = scouting_system.predict_player_role(young_talent)
    print(f"\nPredicted Playing Role: {predicted_role}")
    
    # Find similar players in the database
    print("\nTop 5 Similar Players in Database:")
    similar_players = scouting_system.find_similar_players(young_talent, df_players_with_roles, top_n=5)
    
    for idx, (_, player) in enumerate(similar_players.iterrows(), 1):
        print(f"{idx}. {player['position_role']} "
              f"(Age: {player['edad']}, "
              f"Goles: {player['goles']:.0f}, "
              f"Asist: {player['asistencias']:.0f}, "
              f"Velocidad: {player['velocidad']:.1f} km/h) "
              f"- Similarity: {player['similarity_score']:.3f}")
    
    # Simulate European potential prediction
    # In reality, we'd need historical data of Colombian players who went to Europe
    # For demonstration, we'll create synthetic labels based on performance metrics
    print("\n" + "-"*50)
    print("EUROPEAN POTENTIAL ASSESSMENT")
    print("-"*50)
    
    # Create synthetic labels for European success (based on high performance + youth)
    european_success = (
        (df_players_with_roles['goles'] + df_players_with_roles['asistencias']) > 15  # High productivity
    ).astype(int) * (
        (df_players_with_roles['edad'] < 25)  # Young enough to develop
    ).astype(int) * (
        (df_players_with_roles['minutos'] > 1000)  # Enough playing time
    ).astype(int)
    
    # Add some randomness to make it more realistic
    european_success = np.where(
        np.random.random(n_players) > 0.8,  # 20% chance to flip
        1 - european_success,
        european_success
    )
    
    print(f"Number of players labeled as 'European success': {european_success.sum()}")
    print(f"Number of players labeled as 'Did not succeed in Europe': {(european_success == 0).sum()}")
    
    # Train potential model
    feature_importance = scouting_system.train_potential_model(df_players_with_roles, european_success)
    
    # Predict potential for our young talent
    european_prob = scouting_system.predict_european_potential(young_talent)
    print(f"\nEuropean Success Probability for Sample Talent: {european_prob:.1%}")
    
    # Detect undervalued players
    print("\n" + "-"*50)
    print("UNDERVALUED PLAYER DETECTION")
    print("-"*50)
    
    # Detect undervalued players (using age as proxy for market value)
    undervalued = scouting_system.detect_undervalued_players(df_players_with_roles.copy())
    
    print("Top 5 Most Undervalued Players (High Performance Relative to Age/Cost):")
    for idx, (_, player) in enumerate(undervalued.head().iterrows(), 1):
        print(f"{idx}. {player['position_role']} "
              f"(Age: {player['edad']}, "
              f"Goles: {player['goles']:.0f}, "
              f"Asist: {player['asistencias']:.0f}, "
              f"xG: {player['xG']:.1f}, "
              f"Value Ratio: {player['value_ratio']:.2f})")
    
    # Save models
    print("\n" + "-"*50)
    print("SAVING MODELS")
    print("-"*50)
    scouting_system.save_models('./saved_model')
    
    print("\n" + "="*60)
    print("SCOUTING SYSTEM DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("1. Player clustering by playing style/position")
    print("2. Similarity-based player comparison") 
    print("3. European potential prediction")
    print("4. Undervalued player detection")
    print("\nNext steps for production:")
    print("- Integrate with real data feeds from Colombian leagues")
    print("- Add video analysis and tracking data")
    print("- Incorporate psychological and adaptability factors")
    print("- Develop transfer market valuation models")
    print("- Create mobile scouting applications")