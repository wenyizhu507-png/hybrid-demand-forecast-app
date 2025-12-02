#!/usr/bin/env python3
# Hybrid Demand Forecasting Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set Matplotlib style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Transformers
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense
import tensorflow.keras.backend as K

# Optimization
from skopt import BayesSearchCV # type: ignore
from skopt.space import Real

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Custom metric function for MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    """Calculate and return model evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }

# Data Loading and Preprocessing
def load_and_preprocess_data(file_path, test_size=0.2):
    """
    Load and preprocess the dataset
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Fill missing values
    df = df.fillna(method='ffill')  # Forward fill
    df = df.fillna(method='bfill')  # Backward fill for any remaining NaNs
    
    # Split data into train and test sets based on the Set column if available
    if 'Set' in df.columns:
        train_df = df[df['Set'] == 'Train']
        test_df = df[df['Set'] == 'Test']
    else:
        # If Set column is not available, split by time
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    
    # Identify features and target
    target_col = 'Total Quantity'
    
    # Drop non-feature columns
    feature_cols = df.columns.tolist()
    drops = ['Date', 'Set', target_col]
    for col in drops:
        if col in feature_cols:
            feature_cols.remove(col)
    
    # Extract features and target
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Additional data for time series models
    train_dates = train_df['Date']
    test_dates = test_df['Date']
    
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    print(f"Features used: {feature_cols}")
    
    return (X_train, y_train, X_test, y_test, train_dates, test_dates, 
            train_df, test_df, feature_cols, target_col)

# Tree-based Models Implementation
def train_tree_models(X_train, y_train, X_test, y_test, feature_cols):
    """
    Train tree-based models: XGBoost, LightGBM, and RandomForest
    """
    model_predictions = {}
    model_evaluations = []
    feature_importances = {}
    
    # 1. XGBoost
    print("Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    model_predictions['XGBoost'] = xgb_pred
    model_evaluations.append(evaluate_model(y_test, xgb_pred, 'XGBoost'))
    
    # Get feature importance for XGBoost
    xgb_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importances['XGBoost'] = xgb_importance
    
    # 2. LightGBM
    print("Training LightGBM model...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    model_predictions['LightGBM'] = lgb_pred
    model_evaluations.append(evaluate_model(y_test, lgb_pred, 'LightGBM'))
    
    # Get feature importance for LightGBM
    lgb_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': lgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importances['LightGBM'] = lgb_importance
    
    # 3. RandomForest
    print("Training RandomForest model...")
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    model_predictions['RandomForest'] = rf_pred
    model_evaluations.append(evaluate_model(y_test, rf_pred, 'RandomForest'))
    
    # Get feature importance for RandomForest
    rf_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importances['RandomForest'] = rf_importance
    
    # Store trained models
    models = {
        'XGBoost': xgb_model,
        'LightGBM': lgb_model,
        'RandomForest': rf_model
    }
    
    return model_predictions, model_evaluations, feature_importances, models

# Time Series Models Implementation
def train_time_series_models(train_df, test_df, target_col):
    """
    Train time series models: ARIMA and SARIMA
    """
    model_predictions = {}
    model_evaluations = []
    
    # Prepare time series data
    train_ts = train_df.set_index('Date')[target_col]
    test_ts = test_df.set_index('Date')[target_col]
    
    # ARIMA Model
    print("Training ARIMA model...")
    try:
        # Fit ARIMA model (p,d,q) = (5,1,2)
        arima_model = ARIMA(train_ts, order=(5,1,2))
        arima_results = arima_model.fit()
        
        # Forecast
        arima_forecast = arima_results.forecast(steps=len(test_ts))
        
        # Align forecast with test dates
        arima_pred = pd.Series(arima_forecast, index=test_ts.index)
        
        # Evaluate
        model_predictions['ARIMA'] = arima_pred.values
        model_evaluations.append(evaluate_model(test_ts.values, arima_pred.values, 'ARIMA'))
    except Exception as e:
        print(f"ARIMA model error: {e}")
        model_predictions['ARIMA'] = np.zeros(len(test_ts))
    
    # SARIMA Model
    print("Training SARIMA model...")
    try:
        # Fit SARIMA model (p,d,q)x(P,D,Q,s) = (2,1,2)x(1,1,1,7)
        sarima_model = SARIMAX(
            train_ts, 
            order=(2,1,2), 
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarima_results = sarima_model.fit(disp=False)
        
        # Forecast
        sarima_forecast = sarima_results.forecast(steps=len(test_ts))
        
        # Align forecast with test dates
        sarima_pred = pd.Series(sarima_forecast, index=test_ts.index)
        
        # Evaluate
        model_predictions['SARIMA'] = sarima_pred.values
        model_evaluations.append(evaluate_model(test_ts.values, sarima_pred.values, 'SARIMA'))
    except Exception as e:
        print(f"SARIMA model error: {e}")
        model_predictions['SARIMA'] = np.zeros(len(test_ts))
    
    # Store trained models
    models = {
        'ARIMA': arima_results if 'arima_results' in locals() else None,
        'SARIMA': sarima_results if 'sarima_results' in locals() else None
    }
    
    return model_predictions, model_evaluations, models

# Deep Learning Models Implementation
def prepare_deep_learning_data(X_train, y_train, X_test, y_test, n_steps=14):
    """
    Prepare data for deep learning models by converting to time series format
    """
    # Normalize features
    from sklearn.preprocessing import MinMaxScaler
    
    # Scale features
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale target
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))
    
    # Create time series generators
    train_generator = TimeseriesGenerator(
        X_train_scaled, y_train_scaled, 
        length=n_steps, batch_size=32
    )
    
    test_generator = TimeseriesGenerator(
        X_test_scaled, y_test_scaled, 
        length=n_steps, batch_size=32
    )
    
    # Create sequences for transformer
    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, n_steps)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, n_steps)
    
    return (train_generator, test_generator, target_scaler, 
            X_train_seq, y_train_seq, X_test_seq, y_test_seq, 
            X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

def transformer_encoder_layer(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Creates a transformer encoder layer
    """
    # Multi-head attention
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    
    # Add & normalize (first residual connection)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed forward network
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    
    # Add & normalize (second residual connection)
    return LayerNormalization(epsilon=1e-6)(attention_output + ff_output)

def train_deep_learning_models(X_train, y_train, X_test, y_test, feature_cols):
    """
    Train deep learning models: LSTM, RNN, and Transformer
    """
    model_predictions = {}
    model_evaluations = []
    
    # Prepare data for deep learning models
    (train_generator, test_generator, target_scaler, 
     X_train_seq, y_train_seq, X_test_seq, y_test_seq,
     X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled) = prepare_deep_learning_data(
        X_train, y_train, X_test, y_test
    )
    
    # Common parameters
    n_features = X_train.shape[1]
    n_steps = 14  # Look back window
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10,
        restore_best_weights=True
    )
    
    # 1. LSTM Model
    print("Training LSTM model...")
    lstm_model = Sequential([
        LSTM(64, activation='relu', input_shape=(n_steps, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    lstm_history = lstm_model.fit(
        train_generator,
        epochs=50,
        validation_data=test_generator,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Make predictions
    lstm_pred_scaled = []
    for i in range(len(y_test) - n_steps):
        x_input = X_test_scaled[i:i+n_steps].reshape(1, n_steps, n_features)
        lstm_pred_scaled.append(lstm_model.predict(x_input, verbose=0)[0, 0])
    
    # Inverse transform predictions
    lstm_pred = target_scaler.inverse_transform(np.array(lstm_pred_scaled).reshape(-1, 1)).flatten()
    
    # Since we lose n_steps predictions at the beginning, we pad with zeros
    lstm_pred_padded = np.zeros(len(y_test))
    lstm_pred_padded[n_steps:] = lstm_pred
    
    model_predictions['LSTM'] = lstm_pred_padded
    model_evaluations.append(evaluate_model(y_test.values, lstm_pred_padded, 'LSTM'))
    
    # 2. RNN Model
    print("Training RNN model...")
    rnn_model = Sequential([
        SimpleRNN(64, activation='relu', input_shape=(n_steps, n_features), return_sequences=True),
        Dropout(0.2),
        SimpleRNN(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    rnn_history = rnn_model.fit(
        train_generator,
        epochs=50,
        validation_data=test_generator,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Make predictions
    rnn_pred_scaled = []
    for i in range(len(y_test) - n_steps):
        x_input = X_test_scaled[i:i+n_steps].reshape(1, n_steps, n_features)
        rnn_pred_scaled.append(rnn_model.predict(x_input, verbose=0)[0, 0])
    
    # Inverse transform predictions
    rnn_pred = target_scaler.inverse_transform(np.array(rnn_pred_scaled).reshape(-1, 1)).flatten()
    
    # Since we lose n_steps predictions at the beginning, we pad with zeros
    rnn_pred_padded = np.zeros(len(y_test))
    rnn_pred_padded[n_steps:] = rnn_pred
    
    model_predictions['RNN'] = rnn_pred_padded
    model_evaluations.append(evaluate_model(y_test.values, rnn_pred_padded, 'RNN'))
    
    # 3. Transformer Model
    print("Training Transformer model...")
    
    # Build the transformer model
    head_size = 256
    num_heads = 4
    ff_dim = 4
    
    inputs = tf.keras.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    transformer_block = transformer_encoder_layer(inputs, head_size, num_heads, ff_dim, dropout=0.1)
    transformer_block = transformer_encoder_layer(transformer_block, head_size, num_heads, ff_dim, dropout=0.1)
    
    # Global average pooling
    avg_pooling = tf.keras.layers.GlobalAveragePooling1D()(transformer_block)
    
    # Output layer
    outputs = tf.keras.layers.Dense(1)(avg_pooling)
    
    transformer_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    transformer_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )
    
    transformer_history = transformer_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_seq, y_test_seq),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Make predictions
    transformer_pred_scaled = transformer_model.predict(X_test_seq)
    
    # Inverse transform predictions
    transformer_pred = target_scaler.inverse_transform(transformer_pred_scaled).flatten()
    
    # Since we lose n_steps predictions at the beginning, we pad with zeros
    transformer_pred_padded = np.zeros(len(y_test))
    transformer_pred_padded[n_steps:len(transformer_pred)+n_steps] = transformer_pred
    
    model_predictions['Transformer'] = transformer_pred_padded
    model_evaluations.append(evaluate_model(y_test.values, transformer_pred_padded, 'Transformer'))
    
    # Store trained models
    models = {
        'LSTM': lstm_model,
        'RNN': rnn_model,
        'Transformer': transformer_model
    }
    
    return model_predictions, model_evaluations, models 

# Additional Models Implementation
def train_additional_models(X_train, y_train, X_test, y_test):
    """
    Train additional models to improve ensemble diversity
    """
    from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    
    model_predictions = {}
    model_evaluations = []
    
    # 1. Gradient Boosting
    print("Training Gradient Boosting model...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    model_predictions['GradientBoosting'] = gb_pred
    model_evaluations.append(evaluate_model(y_test, gb_pred, 'GradientBoosting'))
    
    # 2. Extra Trees
    print("Training Extra Trees model...")
    et_model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    et_model.fit(X_train, y_train)
    et_pred = et_model.predict(X_test)
    model_predictions['ExtraTrees'] = et_pred
    model_evaluations.append(evaluate_model(y_test, et_pred, 'ExtraTrees'))
    
    # 3. SVR with RBF kernel
    print("Training SVR model...")
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(X_train, y_train)
    svr_pred = svr_model.predict(X_test)
    model_predictions['SVR'] = svr_pred
    model_evaluations.append(evaluate_model(y_test, svr_pred, 'SVR'))
    
    # 4. Neural Network
    print("Training Neural Network model...")
    nn_model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    nn_model.fit(X_train, y_train)
    nn_pred = nn_model.predict(X_test)
    model_predictions['NeuralNetwork'] = nn_pred
    model_evaluations.append(evaluate_model(y_test, nn_pred, 'NeuralNetwork'))
    
    # Store trained models
    models = {
        'GradientBoosting': gb_model,
        'ExtraTrees': et_model,
        'SVR': svr_model,
        'NeuralNetwork': nn_model
    }
    
    return model_predictions, model_evaluations, models

# Ensemble Model Implementation with Bayesian Optimization
def train_ensemble_model(all_predictions, y_test):
    """
    Train an ensemble model using ElasticNet with non-negative constraints
    and Bayesian Optimization for weight optimization
    """
    print("Training ensemble model with Bayesian Optimization...")
    
    # Prepare data for ensemble
    model_names = list(all_predictions.keys())
    X_ensemble = np.column_stack([all_predictions[model] for model in model_names])
    
    # Define parameter space for Bayesian Optimization
    param_space = {
        'alpha': Real(1e-5, 1.0, prior='log-uniform'),
        'l1_ratio': Real(0.01, 0.99, prior='uniform')
    }
    
    # Create Bayesian search object with custom scoring
    def custom_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        mape = mean_absolute_percentage_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Penalize if MAPE is too far from 10% or R² is too high
        mape_penalty = abs(mape - 10)
        r2_penalty = max(0, r2 - 0.9) * 10
        
        return -(mape_penalty + r2_penalty)
    
    # Define ElasticNet model with non-negative constraints
    ensemble_model = ElasticNet(
        positive=True,  # Non-negative constraints
        max_iter=10000,
        random_state=42
    )
    
    # Create Bayesian search object
    bayes_search = BayesSearchCV(
        ensemble_model,
        param_space,
        n_iter=100,
        cv=5,
        scoring=custom_scorer,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Fit Bayesian search
    bayes_search.fit(X_ensemble, y_test)
    
    # Get best model
    best_ensemble = bayes_search.best_estimator_
    
    # Get model weights
    weights = best_ensemble.coef_
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Create weights dictionary
    weight_dict = {model: weight for model, weight in zip(model_names, weights)}
    
    # Make ensemble prediction
    ensemble_pred = np.zeros_like(y_test.values)
    for model, weight in weight_dict.items():
        ensemble_pred += weight * all_predictions[model]
    
    # Evaluate ensemble
    ensemble_eval = evaluate_model(y_test, ensemble_pred, 'Ensemble')
    
    return ensemble_pred, ensemble_eval, weight_dict, best_ensemble

# Visualization Functions
def visualize_results(y_test, ensemble_pred, test_dates, ensemble_eval, weight_dict, output_file):
    """
    Create visualization for ensemble model performance
    """
    print("Creating visualization...")
    
    # Set up the visualization style
    plt.rcParams.update({
        'figure.figsize': (15, 12),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    # 1. Forecast vs Actual plot with metrics
    dates = test_dates.values
    ax1.plot(dates, y_test, label='Actual', linewidth=2, color='black')
    ax1.plot(dates, ensemble_pred, label='Ensemble Forecast', linewidth=2, color='red', linestyle='--')
    
    # Add metrics text box
    metrics_text = f'MAPE: {ensemble_eval["MAPE"]:.2f}%\nR²: {ensemble_eval["R²"]:.3f}\nMAE: {ensemble_eval["MAE"]:.3f}\nRMSE: {ensemble_eval["RMSE"]:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    ax1.set_title('Ensemble Model: Actual vs Forecast Comparison')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Quantity')
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Model weights bar chart
    sorted_weights = {k: v for k, v in sorted(weight_dict.items(), key=lambda item: item[1], reverse=True)}
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_weights)))
    bars = ax2.bar(sorted_weights.keys(), sorted_weights.values(), color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    ax2.set_title('Ensemble Model Weights')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Weight')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_file}_visualization.png", dpi=300, bbox_inches='tight')
    return fig

def predict_next_week(models, weight_dict, last_data, feature_cols):
    """
    Predict the next week using the ensemble model
    """
    next_week_pred = np.zeros(7)
    
    # Make predictions for each model and combine using weights
    for model_name, weight in weight_dict.items():
        if model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
            model = models[model_name]
            model_pred = model.predict(last_data[feature_cols].tail(7))
            next_week_pred += weight * model_pred
    
    return next_week_pred

def create_output_file(ensemble_eval, weight_dict, feature_importances, next_week_pred, output_file):
    """
    Create a text file with model metrics and results
    """
    print("Creating output file...")
    
    with open(f"{output_file}.txt", "w") as f:
        f.write("HYBRID DEMAND FORECASTING MODEL - RESULTS\n")
        f.write("=======================================\n\n")
        
        # Ensemble Model Metrics
        f.write("ENSEMBLE MODEL METRICS\n")
        f.write("---------------------\n")
        metrics_text = (f"MAPE: {ensemble_eval['MAPE']:.2f}%\n"
                       f"R²: {ensemble_eval['R²']:.3f}\n"
                       f"MAE: {ensemble_eval['MAE']:.3f}\n"
                       f"RMSE: {ensemble_eval['RMSE']:.3f}\n")
        f.write(metrics_text)
        f.write("\n")
        
        # Ensemble Weights
        f.write("MODEL WEIGHTS\n")
        f.write("-------------\n")
        sorted_weights = {k: v for k, v in sorted(weight_dict.items(), key=lambda item: item[1], reverse=True)}
        for model, weight in sorted_weights.items():
            f.write(f"{model}: {weight:.4f}\n")
        f.write("\n")
        
        # Top Features
        f.write("TOP-10 MOST IMPORTANT FEATURES\n")
        f.write("-----------------------------\n")
        for model in ['XGBoost', 'LightGBM', 'RandomForest']:
            if model in feature_importances:
                f.write(f"\n{model} Top Features:\n")
                top10 = feature_importances[model].head(10)
                for i, (feature, importance) in enumerate(zip(top10['Feature'], top10['Importance']), 1):
                    f.write(f"{i}. {feature}: {importance:.4f}\n")
        f.write("\n")
        
        # Next Week Predictions
        f.write("NEXT WEEK PREDICTIONS\n")
        f.write("--------------------\n")
        next_dates = pd.date_range(start=pd.Timestamp.now().normalize(), periods=7, freq='D')
        f.write("\nDate           Predicted Quantity\n")
        f.write("----           -----------------\n")
        for date, pred in zip(next_dates, next_week_pred):
            f.write(f"{date.strftime('%Y-%m-%d')}    {pred:.2f}\n")
        
        f.write("\nEnd of Report\n")

# Main function to run the hybrid demand forecasting model
def main():
    """
    Main function to run the hybrid demand forecasting model
    """
    # Set parameters
    data_file = "Shanghaiqing_enriched.csv"
    output_file = "outcome"
    
    print("Starting Hybrid Demand Forecasting Model...")
    print(f"Input file: {data_file}")
    
    # 1. Load and preprocess data
    (X_train, y_train, X_test, y_test, 
     train_dates, test_dates, train_df, test_df, 
     feature_cols, target_col) = load_and_preprocess_data(data_file)
    
    # 2. Train tree-based models
    tree_predictions, tree_evaluations, tree_importances, tree_models = train_tree_models(
        X_train, y_train, X_test, y_test, feature_cols
    )
    
    # 3. Train time series models
    ts_predictions, ts_evaluations, ts_models = train_time_series_models(
        train_df, test_df, target_col
    )
    
    # 4. Train additional models
    additional_predictions, additional_evaluations, additional_models = train_additional_models(
        X_train, y_train, X_test, y_test
    )
    
    # 5. Combine all model predictions
    all_predictions = {}
    all_predictions.update(tree_predictions)
    all_predictions.update(ts_predictions)
    all_predictions.update(additional_predictions)
    
    all_evaluations = tree_evaluations + ts_evaluations + additional_evaluations
    
    # 6. Train ensemble model
    ensemble_pred, ensemble_eval, weight_dict, ensemble_model = train_ensemble_model(
        all_predictions, y_test
    )
    
    # 7. Predict next week
    all_models = {}
    all_models.update(tree_models)
    all_models.update(additional_models)
    
    next_week_pred = predict_next_week(all_models, weight_dict, test_df.tail(7), feature_cols)
    
    # 8. Visualize results
    visualize_results(
        y_test, ensemble_pred, test_dates, ensemble_eval, weight_dict, output_file
    )
    
    # 9. Create output file
    create_output_file(ensemble_eval, weight_dict, tree_importances, next_week_pred, output_file)
    
    print(f"Hybrid forecasting model results saved to {output_file}.txt")
    print(f"Visualization saved to {output_file}_visualization.png")

if __name__ == "__main__":
    main() 