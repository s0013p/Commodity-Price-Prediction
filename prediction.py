import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class SarimaXGBoostEnsembleModel:
    def __init__(self):
        self.sarima_model = None
        self.sarima_fit = None
        self.xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8, 
                                     colsample_bytree=0.8, random_state=42, objective='reg:squarederror', eval_metric='rmse')
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_columns = ['Min_Price', 'Max_Price']
        self.target_column = 'Modal_Price'
        self.date_column = 'Arrival_Date'
        self.features_importance = {}
        self.ensemble_weights = {'sarima': 0.6, 'xgboost': 0.4}
        
    def preprocess_data(self, df):
        """Preprocess the data for modeling."""
        try:
            data = df.copy()
            data[self.date_column] = pd.to_datetime(data[self.date_column])
            data = data.sort_values(by=self.date_column)
            
            # Convert price columns to numeric
            for col in [self.target_column] + self.feature_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            data = data.dropna(subset=[self.target_column])
            
            # Create features
            data['price_ratio'] = data['Max_Price'] / data['Min_Price']
            data['price_diff'] = data['Max_Price'] - data['Min_Price']
            
            # Include weather and CPI features if available
            for feature in ['temperature', 'visibility', 'wind_speed', 'clouds', 'CPI']:
                if feature in data.columns:
                    data[feature] = pd.to_numeric(data[feature], errors='coerce')
                    self.feature_columns.append(feature)
            
            # Fill missing values
            for col in self.feature_columns:
                if col in data.columns:
                    data[col] = data[col].ffill().bfill()
            
            # Time-based features
            data['month'] = data[self.date_column].dt.month
            data['day_of_week'] = data[self.date_column].dt.dayofweek
            data['day_of_year'] = data[self.date_column].dt.dayofyear
            
            # Lag and rolling features
            for lag in [1, 3, 7]:
                data[f'price_lag_{lag}'] = data[self.target_column].shift(lag)
            for window in [3, 7, 14]:
                data[f'price_rolling_mean_{window}'] = data[self.target_column].rolling(window=window).mean()
                data[f'price_rolling_std_{window}'] = data[self.target_column].rolling(window=window).std()
            
            return data.dropna()
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None
    
    def train(self, df):
        """Train the ensemble model."""
        try:
            data = self.preprocess_data(df)
            if data is None or len(data) < 30:
                print("Insufficient data for training")
                return False
            
            train_size = int(len(data) * 0.8)
            train_data, val_data = data.iloc[:train_size], data.iloc[train_size:]
            
            # Train SARIMA
            self.sarima_model = SARIMAX(train_data[self.target_column], order=(1, 1, 1), 
                                       seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
            self.sarima_fit = self.sarima_model.fit(disp=False)
            
            # Prepare XGBoost features
            feature_list = self.feature_columns + ['price_ratio', 'price_diff', 'month', 'day_of_week', 
                                                  'price_lag_1', 'price_lag_3', 'price_lag_7', 'price_rolling_mean_7', 'price_rolling_std_7']
            X_train, y_train = train_data[feature_list], train_data[self.target_column]
            X_val, y_val = val_data[feature_list], val_data[self.target_column]
            
            # Scale and train XGBoost
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
            self.xgb_model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)
            
            self.features_importance = dict(zip(X_train.columns, self.xgb_model.feature_importances_))
            self._optimize_weights(val_data)
            self.trained = True
            return True
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return False

    def _optimize_weights(self, val_data):
        """Optimize ensemble weights."""
        try:
            feature_list = self.feature_columns + ['price_ratio', 'price_diff', 'month', 'day_of_week', 
                                                  'price_lag_1', 'price_lag_3', 'price_lag_7', 'price_rolling_mean_7', 'price_rolling_std_7']
            X_val_scaled = self.scaler.transform(val_data[feature_list])
            sarima_pred = self.sarima_fit.forecast(steps=len(val_data))
            xgb_pred = self.xgb_model.predict(X_val_scaled)
            
            best_mape, best_weights = float('inf'), {'sarima': 0.5, 'xgboost': 0.5}
            for sarima_weight in np.arange(0.1, 1.0, 0.1):
                xgb_weight = 1 - sarima_weight
                ensemble_pred = sarima_weight * sarima_pred + xgb_weight * xgb_pred
                mape = mean_absolute_percentage_error(val_data[self.target_column], ensemble_pred) * 100
                if mape < best_mape:
                    best_mape, best_weights = mape, {'sarima': sarima_weight, 'xgboost': xgb_weight}
            self.ensemble_weights = best_weights
        except Exception as e:
            print(f"Error in weight optimization: {str(e)}")
            self.ensemble_weights = {'sarima': 0.6, 'xgboost': 0.4}
    
    def predict(self, prediction_date_str):
        """Generate predictions for 5 days ahead."""
        if not self.trained:
            print("Model is not trained yet")
            return pd.DataFrame()
        
        try:
            prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d')
            future_dates = [prediction_date + timedelta(days=i) for i in range(5)]
            predictions_df = pd.DataFrame({'date': future_dates, 'predicted_price': 0.0, 'lower_bound': 0.0, 'upper_bound': 0.0})
            
            # SARIMA forecast
            sarima_mean = self.sarima_fit.forecast(steps=5)
            sarima_mean_values = sarima_mean.values if hasattr(sarima_mean, 'values') else np.array(sarima_mean)
            sarima_conf_int = self.sarima_fit.get_forecast(steps=5).conf_int(alpha=0.05)
            
            # XGBoost predictions
            feature_means = self.scaler.mean_[:len(self.feature_columns)]
            xgb_predictions = []
            
            for i, future_date in enumerate(future_dates):
                feature_dict = {col: feature_means[j] for j, col in enumerate(self.feature_columns)}
                feature_dict.update({
                    'price_ratio': feature_dict['Max_Price'] / feature_dict['Min_Price'] if feature_dict['Min_Price'] != 0 else 1.0,
                    'price_diff': feature_dict['Max_Price'] - feature_dict['Min_Price'],
                    'month': future_date.month, 'day_of_week': future_date.weekday(),
                    'day_of_year': future_date.timetuple().tm_yday
                })
                
                # Lag and rolling features
                if i == 0:
                    for lag in [1, 3, 7]:
                        feature_dict[f'price_lag_{lag}'] = sarima_mean_values[0]
                    feature_dict['price_rolling_mean_7'] = sarima_mean_values[0]
                    feature_dict['price_rolling_std_7'] = 0.1 * sarima_mean_values[0]
                else:
                    feature_dict['price_lag_1'] = xgb_predictions[i-1]
                    feature_dict['price_lag_3'] = xgb_predictions[i-3] if i >= 3 else sarima_mean_values[0]
                    feature_dict['price_lag_7'] = xgb_predictions[i-7] if i >= 7 else sarima_mean_values[0]
                    prev_preds = xgb_predictions[:i]
                    feature_dict['price_rolling_mean_7'] = np.mean(prev_preds[-7:] if len(prev_preds) >= 7 else prev_preds + [sarima_mean_values[0]] * (7 - len(prev_preds)))
                    feature_dict['price_rolling_std_7'] = np.std(prev_preds[-7:]) if len(prev_preds) >= 7 else 0.1 * feature_dict['price_rolling_mean_7']
                
                # Prepare and scale features
                feature_list = [feature_dict.get(col, 0.0) for col in self.feature_columns + ['price_ratio', 'price_diff', 'month', 'day_of_week', 'price_lag_1', 'price_lag_3', 'price_lag_7', 'price_rolling_mean_7', 'price_rolling_std_7']]
                scaled_features = self.scaler.transform([feature_list])
                xgb_predictions.append(self.xgb_model.predict(scaled_features)[0])
            
            # Combine predictions
            for i in range(5):
                ensemble_pred = (self.ensemble_weights['sarima'] * sarima_mean_values[i] + 
                               self.ensemble_weights['xgboost'] * xgb_predictions[i])
                lower_bound = sarima_conf_int.iloc[i, 0] * self.ensemble_weights['sarima'] + xgb_predictions[i] * (1 - 0.1) * self.ensemble_weights['xgboost']
                upper_bound = sarima_conf_int.iloc[i, 1] * self.ensemble_weights['sarima'] + xgb_predictions[i] * (1 + 0.1) * self.ensemble_weights['xgboost']
                
                predictions_df.loc[i, 'predicted_price'] = ensemble_pred
                predictions_df.loc[i, 'lower_bound'] = lower_bound
                predictions_df.loc[i, 'upper_bound'] = upper_bound
            
            return predictions_df
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()

    def backtesting(self, df):
        """Perform backtesting on historical data."""
        try:
            data = self.preprocess_data(df)
            if data is None or len(data) < 30:
                print("Insufficient data for backtesting")
                return None
            
            test_size = int(len(data) * 0.2)
            train_data, test_data = data.iloc[:-test_size], data.iloc[-test_size:]
            
            # Train models
            sarima_model = SARIMAX(train_data[self.target_column], order=(1, 1, 1), 
                                  seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
            sarima_fit = sarima_model.fit(disp=False)
            sarima_pred = sarima_fit.forecast(steps=len(test_data))
            
            feature_list = self.feature_columns + ['price_ratio', 'price_diff', 'month', 'day_of_week', 
                                                  'price_lag_1', 'price_lag_3', 'price_lag_7', 'price_rolling_mean_7', 'price_rolling_std_7']
            X_train, y_train = train_data[feature_list], train_data[self.target_column]
            X_test, y_test = test_data[feature_list], test_data[self.target_column]
            
            scaler = StandardScaler()
            X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
            
            xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8, 
                                   colsample_bytree=0.8, random_state=42, objective='reg:squarederror', eval_metric='rmse')
            xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train)], verbose=False)
            xgb_pred = xgb_model.predict(X_test_scaled)
            
            # Ensemble predictions and metrics
            ensemble_pred = (self.ensemble_weights['sarima'] * sarima_pred + self.ensemble_weights['xgboost'] * xgb_pred)
            mean_actual_price = np.mean(y_test)
            
            return {
                'test_dates': test_data[self.date_column],
                'actual_prices': y_test.values,
                'sarima_predictions': sarima_pred,
                'xgb_predictions': xgb_pred,
                'ensemble_predictions': ensemble_pred,
                'metrics': {
                    'rmse': round((np.sqrt(mean_squared_error(y_test, ensemble_pred)) / mean_actual_price) * 100, 2),
                    'mae': round((mean_absolute_error(y_test, ensemble_pred) / mean_actual_price) * 100, 2),
                    'r2': round(r2_score(y_test, ensemble_pred), 4),
                    'mape': round(mean_absolute_percentage_error(y_test, ensemble_pred) * 100, 2)
                }
            }
        except Exception as e:
            print(f"Error in backtesting: {str(e)}")
            return None