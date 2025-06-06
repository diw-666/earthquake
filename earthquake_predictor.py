import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

class EarthquakePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        # Base date for day conversion (assuming days since 1900-01-01)
        self.base_date = datetime(1900, 1, 1)
        
    def days_to_date(self, days):
        """Convert days since base date to actual calendar date"""
        try:
            actual_date = self.base_date + timedelta(days=float(days))
            return actual_date.strftime('%Y-%m-%d')
        except:
            return f"Day {days:.1f}"
        
    def load_data(self, filepath='data/Earthquakes_South_Asia_Cleaned.csv'):
        """Load and preprocess earthquake data"""
        print("Loading earthquake data...")
        df = pd.read_csv(filepath)
        
        # Basic data info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['days'].min():.1f} to {df['days'].max():.1f} days")
        
        return df
    
    def engineer_features(self, df):
        """Create additional features for better prediction"""
        print("Engineering features...")
        
        # Sort by time
        df_sorted = df.sort_values('days').copy()
        
        # Time-based features
        df_sorted['year'] = (df_sorted['days'] / 365.25).astype(int) + 1900  # Assuming days since 1900
        df_sorted['day_of_year'] = df_sorted['days'] % 365.25
        
        # Spatial clustering features
        df_sorted['region_x'] = (df_sorted['x'] // 5) * 5  # 5-degree bins
        df_sorted['region_y'] = (df_sorted['y'] // 5) * 5
        
        # Historical features (rolling windows)
        for window in [30, 90, 365]:
            # Count of earthquakes in time window
            df_sorted[f'count_last_{window}d'] = df_sorted.groupby(['region_x', 'region_y'])['mag'].transform(
                lambda x: x.rolling(window=min(len(x), window), min_periods=1).count()
            )
            
            # Average magnitude in time window
            df_sorted[f'avg_mag_last_{window}d'] = df_sorted.groupby(['region_x', 'region_y'])['mag'].transform(
                lambda x: x.rolling(window=min(len(x), window), min_periods=1).mean()
            )
            
            # Maximum magnitude in time window
            df_sorted[f'max_mag_last_{window}d'] = df_sorted.groupby(['region_x', 'region_y'])['mag'].transform(
                lambda x: x.rolling(window=min(len(x), window), min_periods=1).max()
            )
        
        # Distance from major fault lines (simplified - using high activity areas)
        high_activity_regions = df_sorted.groupby(['region_x', 'region_y'])['mag'].count().sort_values(ascending=False).head(10)
        df_sorted['near_fault'] = 0
        for (rx, ry), count in high_activity_regions.items():
            distance = np.sqrt((df_sorted['region_x'] - rx)**2 + (df_sorted['region_y'] - ry)**2)
            df_sorted['near_fault'] += 1 / (1 + distance)
        
        # Time since last major earthquake (>6.0)
        major_eq_mask = df_sorted['mag'] >= 6.0
        df_sorted['days_since_major'] = np.nan
        
        for region in df_sorted[['region_x', 'region_y']].drop_duplicates().values:
            rx, ry = region
            region_mask = (df_sorted['region_x'] == rx) & (df_sorted['region_y'] == ry)
            region_data = df_sorted[region_mask].copy()
            
            for idx in region_data.index:
                current_day = df_sorted.loc[idx, 'days']
                major_before = region_data[(region_data['days'] < current_day) & 
                                         (region_data['mag'] >= 6.0)]
                if len(major_before) > 0:
                    last_major_day = major_before['days'].max()
                    df_sorted.loc[idx, 'days_since_major'] = current_day - last_major_day
                else:
                    df_sorted.loc[idx, 'days_since_major'] = 9999  # No previous major earthquake
        
        # Fill remaining NaN values
        df_sorted['days_since_major'].fillna(9999, inplace=True)
        
        return df_sorted
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        feature_columns = [
            'depth', 'x', 'y', 'day_of_year', 'region_x', 'region_y',
            'count_last_30d', 'count_last_90d', 'count_last_365d',
            'avg_mag_last_30d', 'avg_mag_last_90d', 'avg_mag_last_365d',
            'max_mag_last_30d', 'max_mag_last_90d', 'max_mag_last_365d',
            'near_fault', 'days_since_major'
        ]
        
        X = df[feature_columns].copy()
        y = df['mag'].copy()
        
        # Handle any remaining NaN values
        X.fillna(X.median(), inplace=True)
        
        self.feature_names = feature_columns
        return X, y
    
    def train_models(self, X, y):
        """Train Gradient Boosting model for earthquake prediction"""
        print("Training Gradient Boosting model...")
        
        # Split data (time series aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Train only Gradient Boosting
        print("Training Gradient Boosting...")
        gb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 10, 15],
            'learning_rate': [0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0]
        }
        gb = GradientBoostingRegressor(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gb_grid.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_grid.best_estimator_
        
        print(f"Best parameters: {gb_grid.best_params_}")
        
        # Evaluate model
        model = self.models['gradient_boosting']
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.evaluation_results = {
            'gradient_boosting': {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
        }
        
        print(f"Gradient Boosting: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        
        self.is_trained = True
        return X_test, y_test
    
    def predict_next_earthquakes(self, df, days_ahead=30):
        """Predict earthquakes in the next specified days"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        # Get the latest data point for each region
        latest_data = df.groupby(['region_x', 'region_y']).last().reset_index()
        
        predictions = []
        
        for _, region_data in latest_data.iterrows():
            # Create future time points
            current_day = region_data['days']
            future_days = np.linspace(current_day + 1, current_day + days_ahead, days_ahead)
            
            for future_day in future_days:
                # Create features for future prediction
                future_features = {
                    'depth': region_data['depth'],  # Use historical average
                    'x': region_data['x'],
                    'y': region_data['y'],
                    'day_of_year': future_day % 365.25,
                    'region_x': region_data['region_x'],
                    'region_y': region_data['region_y'],
                    'count_last_30d': region_data['count_last_30d'],
                    'count_last_90d': region_data['count_last_90d'],
                    'count_last_365d': region_data['count_last_365d'],
                    'avg_mag_last_30d': region_data['avg_mag_last_30d'],
                    'avg_mag_last_90d': region_data['avg_mag_last_90d'],
                    'avg_mag_last_365d': region_data['avg_mag_last_365d'],
                    'max_mag_last_30d': region_data['max_mag_last_30d'],
                    'max_mag_last_90d': region_data['max_mag_last_90d'],
                    'max_mag_last_365d': region_data['max_mag_last_365d'],
                    'near_fault': region_data['near_fault'],
                    'days_since_major': future_day - current_day + region_data['days_since_major']
                }
                
                # Convert to DataFrame
                future_df = pd.DataFrame([future_features])
                future_df = future_df[self.feature_names]
                
                # Get prediction from Gradient Boosting model
                model = self.models['gradient_boosting']
                predicted_magnitude = model.predict(future_df)[0]
                
                # Only include predictions above certain threshold
                if predicted_magnitude >= 4.0:  # Minimum magnitude threshold
                    predictions.append({
                        'predicted_magnitude': predicted_magnitude,
                        'latitude': region_data['y'],
                        'longitude': region_data['x'],
                        'predicted_day': future_day,
                        'predicted_date': self.days_to_date(future_day),
                        'region_x': region_data['region_x'],
                        'region_y': region_data['region_y'],
                        'risk_level': self._get_risk_level(predicted_magnitude)
                    })
        
        return pd.DataFrame(predictions)
    
    def _get_risk_level(self, magnitude):
        """Classify risk level based on magnitude"""
        if magnitude >= 7.0:
            return "Very High"
        elif magnitude >= 6.0:
            return "High"
        elif magnitude >= 5.0:
            return "Medium"
        else:
            return "Low"
    
    def save_models(self, filepath='models/'):
        """Save trained model"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        # Save only Gradient Boosting model
        joblib.dump(self.models['gradient_boosting'], f"{filepath}gradient_boosting_model.pkl")
        
        joblib.dump(self.scalers, f"{filepath}scalers.pkl")
        joblib.dump(self.feature_names, f"{filepath}feature_names.pkl")
        
        print(f"Gradient Boosting model saved to {filepath}")
    
    def load_models(self, filepath='models/'):
        """Load pre-trained model"""
        import os
        
        if not os.path.exists(filepath):
            print("No saved models found")
            return False
        
        try:
            # Load only Gradient Boosting model
            if os.path.exists(f"{filepath}gradient_boosting_model.pkl"):
                self.models['gradient_boosting'] = joblib.load(f"{filepath}gradient_boosting_model.pkl")
            else:
                print("Gradient Boosting model not found")
                return False
            
            self.scalers = joblib.load(f"{filepath}scalers.pkl")
            self.feature_names = joblib.load(f"{filepath}feature_names.pkl")
            self.is_trained = True
            
            print("Gradient Boosting model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Main function to train the earthquake prediction model"""
    predictor = EarthquakePredictor()
    
    # Load and process data
    df = predictor.load_data()
    df_featured = predictor.engineer_features(df)
    X, y = predictor.prepare_features(df_featured)
    
    # Train models
    X_test, y_test = predictor.train_models(X, y)
    
    # Save models
    predictor.save_models()
    
    # Generate predictions
    predictions = predictor.predict_next_earthquakes(df_featured, days_ahead=30)
    print(f"\nGenerated {len(predictions)} earthquake predictions for the next 30 days")
    
    # Save predictions
    predictions.to_csv('earthquake_predictions.csv', index=False)
    
    # Display evaluation results
    print("\nGradient Boosting Model Performance:")
    for model_name, metrics in predictor.evaluation_results.items():
        print(f"{model_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
    
    return predictor, predictions

if __name__ == "__main__":
    predictor, predictions = main() 