import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
def load_and_preprocess_data(excel_file_path):
    # Load the data
    print("Reading Excel file...")
    df = pd.read_excel(excel_file_path)
    
    # Display initial columns and data types for debugging
    print("DataFrame columns:", df.columns.tolist())
    print("DataFrame dtypes sample:", df.dtypes.head())
    
    # Convert date column to datetime if it exists and is not already datetime
    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_dtype(df['Date']):
            print("Converting Date to datetime...")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        print("Warning: Date column not found. Using index as proxy for time.")
        df['Date'] = pd.date_range(start='2023-01-01', periods=len(df))
    
    # Extract scores from Results column - with robust error handling
    if 'Results' in df.columns and df['Results'].dtype == 'object':
        print("Processing Results column...")
        # Check if Results column follows the expected format
        if df['Results'].str.contains('-').any():
            try:
                # Extract Win/Loss information if format is like "W 3-2"
                if df['Results'].str.contains(' ').any():
                    df['Win_Loss'] = df['Results'].str.split(' ', n=1).str[0]
                    df['Score'] = df['Results'].str.split(' ', n=1).str[1]
                else:
                    df['Score'] = df['Results']
                
                # Extract score components
                df['Team_Score'] = df['Score'].str.split('-').str[0].astype(int)
                df['Opponent_Score'] = df['Score'].str.split('-').str[1].astype(int)
            except Exception as e:
                print(f"Error parsing Results column: {e}")
                # Use GF and GA as fallback
                print("Using GF and GA columns as direct score indicators.")
                df['Team_Score'] = df['GF'] if 'GF' in df.columns else 0
                df['Opponent_Score'] = df['GA'] if 'GA' in df.columns else 0
        else:
            print("Results column doesn't contain expected format. Using GF and GA columns.")
            df['Team_Score'] = df['GF'] if 'GF' in df.columns else 0
            df['Opponent_Score'] = df['GA'] if 'GA' in df.columns else 0
    else:
        print("Results column not found or not string type. Using GF and GA columns directly.")
        # Use GF and GA columns directly if they exist
        df['Team_Score'] = df['GF'] if 'GF' in df.columns else 0
        df['Opponent_Score'] = df['GA'] if 'GA' in df.columns else 0
    
    # Create features for time trends (days since season start)
    season_start = df['Date'].min()
    df['Days_Since_Season_Start'] = (df['Date'] - season_start).dt.days
    
    # Create team form features (rolling averages)
    print("Creating team rolling statistics...")
    
    # Ensure Team column exists
    if 'Team' not in df.columns:
        print("Warning: Team column not found. Using first column as team identifier.")
        df['Team'] = df.iloc[:, 0]
    
    teams = df['Team'].unique()
    print(f"Found {len(teams)} unique teams.")
    
    # Create empty dataframes to store team stats
    team_rolling_stats = pd.DataFrame()
    
    # Define columns that should be available for rolling stats
    stat_columns = [
        'GF', 'GA', 'TEAM SOG', 'OPPONENT SOG', 'TEAM PPG', 'TEAM PIM',
        'CF%', 'FF%', 'FO%', 'PDO'
    ]
    
    # Check which columns actually exist in the dataframe
    available_columns = [col for col in stat_columns if col in df.columns]
    print(f"Available stat columns: {available_columns}")
    
    if len(available_columns) < 3:
        print("Warning: Few stat columns available. Will use basic scoring stats.")
        
        # If key columns missing, create some basic ones from available data
        if 'GF' not in df.columns and 'Team_Score' in df.columns:
            df['GF'] = df['Team_Score']
        if 'GA' not in df.columns and 'Opponent_Score' in df.columns:
            df['GA'] = df['Opponent_Score']
        
        # Update available columns list
        available_columns = [col for col in stat_columns if col in df.columns]
    
    for team in teams:
        team_data = df[df['Team'] == team].sort_values('Date')
        
        if len(team_data) < 2:
            print(f"Warning: Team {team} has very few data points. Skipping rolling statistics.")
            continue
        
        # Calculate rolling averages for available stats (last 5 games)
        window = min(5, len(team_data) - 1)  # Ensure window isn't larger than data
        
        # Create a dictionary for rolling stats
        rolling_dict = {
            'Team': team,
            'Date': team_data['Date']
        }
        
        # Add rolling statistics for each available column
        for col in available_columns:
            if col in team_data.columns and pd.api.types.is_numeric_dtype(team_data[col]):
                col_name = f"Rolling_{col.replace('%', 'Pct').replace(' ', '_')}"
                rolling_dict[col_name] = team_data[col].rolling(window, min_periods=1).mean()
        
        # Create DataFrame from the dictionary
        rolling_stats = pd.DataFrame(rolling_dict)
        
        # Concatenate to the main rolling stats DataFrame
        team_rolling_stats = pd.concat([team_rolling_stats, rolling_stats], ignore_index=True)
    
    # Check if we have rolling stats before merging
    if len(team_rolling_stats) > 0:
        print(f"Merging {len(team_rolling_stats)} rows of rolling statistics.")
        # Merge rolling stats back to the main dataframe
        df = pd.merge(df, team_rolling_stats, on=['Team', 'Date'], how='left')
    else:
        print("Warning: No rolling statistics were generated. Using static features only.")
    
    # Fill NaN values
    print("Handling missing values...")
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    
    return df

# Step 2: Feature engineering
def engineer_features(df):
    print("Engineering features...")
    
    # Identify the rolling stats columns that were created
    rolling_cols = [col for col in df.columns if col.startswith('Rolling_')]
    print(f"Available rolling stat features: {rolling_cols}")
    
    # If we have at least 3 rolling features, use them; otherwise, use basic features
    if len(rolling_cols) >= 3:
        features = rolling_cols
    else:
        # Fallback to basic features
        print("Using basic features due to limited rolling statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target variables and unnecessary columns
        features = [col for col in numeric_cols if col not in ['Team_Score', 'Opponent_Score', 'index']]
    
    # Always include time feature if available
    if 'Days_Since_Season_Start' in df.columns:
        if 'Days_Since_Season_Start' not in features:
            features.append('Days_Since_Season_Start')
    
    print(f"Selected {len(features)} features: {features[:min(5, len(features))]}...")
    
    # Add team encoding
    print("Encoding team names...")
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    
    # Reshape to ensure 2D array for encoding
    team_data = df[['Team']].values
    team_encoded = encoder.fit_transform(team_data)
    team_encoded_df = pd.DataFrame(
        team_encoded, 
        columns=[f'Team_{i}' for i in range(team_encoded.shape[1])]
    )
    
    # Prepare feature matrix
    X = pd.concat([df[features].reset_index(drop=True), team_encoded_df.reset_index(drop=True)], axis=1)
    
    # Target variables: Team score and opponent score
    y_team = df['Team_Score']
    y_opponent = df['Opponent_Score']
    
    return X, y_team, y_opponent, encoder, features

# Step 3: Build and train the neural network model
def build_and_train_model(X, y_team, y_opponent):
    print("Preparing data for model training...")
    # Split the data into training and testing sets
    X_train, X_test, y_team_train, y_team_test, y_opponent_train, y_opponent_test = train_test_split(
        X, y_team, y_opponent, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert target variables to numpy arrays
    y_team_train = np.array(y_team_train)
    y_team_test = np.array(y_team_test)
    y_opponent_train = np.array(y_opponent_train)
    y_opponent_test = np.array(y_opponent_test)
    
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Feature count: {X_train_scaled.shape[1]}")
    
    # Build the model
    print("Building neural network model...")
    inputs = keras.Input(shape=(X_train_scaled.shape[1],))
    
    # Start with more nodes and gradually reduce
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Two output heads for team score and opponent score
    team_score_output = layers.Dense(1, name='team_score')(x)
    opponent_score_output = layers.Dense(1, name='opponent_score')(x)
    
    model = keras.Model(inputs=inputs, outputs=[team_score_output, opponent_score_output])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={'team_score': 'mse', 'opponent_score': 'mse'},
        metrics={'team_score': 'mae', 'opponent_score': 'mae'}
    )
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=15, 
            restore_best_weights=True,
            monitor='val_loss'
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Train the model
    print("Training the model (this may take a while)...")
    history = model.fit(
        X_train_scaled,
        {'team_score': y_team_train, 'opponent_score': y_opponent_train},
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    print("Evaluating model performance...")
    eval_results = model.evaluate(
        X_test_scaled,
        {'team_score': y_team_test, 'opponent_score': y_opponent_test},
        verbose=0
    )
    
    print(f"Test Loss (Team Score): {eval_results[1]:.4f}")
    print(f"Test MAE (Team Score): {eval_results[3]:.4f}")
    print(f"Test Loss (Opponent Score): {eval_results[2]:.4f}")
    print(f"Test MAE (Opponent Score): {eval_results[4]:.4f}")
    
    # Plot training history
    print("Generating training history plots...")
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history.history['team_score_loss'])
    plt.plot(history.history['val_team_score_loss'])
    plt.title('Team Score Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['team_score_mae'])
    plt.plot(history.history['val_team_score_mae'])
    plt.title('Team Score MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(2, 2, 3)
    plt.plot(history.history['opponent_score_loss'])
    plt.plot(history.history['val_opponent_score_loss'])
    plt.title('Opponent Score Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(2, 2, 4)
    plt.plot(history.history['opponent_score_mae'])
    plt.plot(history.history['val_opponent_score_mae'])
    plt.title('Opponent Score MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model, scaler, history

# Step 4: Make predictions for future games
def predict_game_score(model, scaler, encoder, features, team, opponent_stats, days_since_season_start):
    print(f"Generating prediction for {team}...")
    # Create a dataframe for the prediction
    pred_data = {
        'Team': [team],
        'Days_Since_Season_Start': [days_since_season_start]
    }
    
    # Add stats from the opponent_stats dictionary
    for feature in features:
        if feature in opponent_stats:
            pred_data[feature] = [opponent_stats[feature]]
        else:
            # If a feature is missing, add a default value
            print(f"Warning: Feature {feature} not provided. Using default value.")
            pred_data[feature] = [0.0]
    
    team_df = pd.DataFrame(pred_data)
    
    # Encode the team
    team_encoded = encoder.transform(team_df[['Team']])
    team_encoded_df = pd.DataFrame(
        team_encoded, 
        columns=[f'Team_{i}' for i in range(team_encoded.shape[1])]
    )
    
    # Get only the features that were used during training
    X_pred_features = pd.DataFrame()
    for feature in features:
        if feature in team_df.columns:
            X_pred_features[feature] = team_df[feature]
    
    # Combine features
    X_pred = pd.concat([X_pred_features.reset_index(drop=True), team_encoded_df.reset_index(drop=True)], axis=1)
    
    # Scale features
    X_pred_scaled = scaler.transform(X_pred)
    
    # Make prediction
    predictions = model.predict(X_pred_scaled)
    
    # Round to nearest integer and ensure non-negative
    team_score = max(0, round(float(predictions[0][0])))
    opponent_score = max(0, round(float(predictions[1][0])))
    
    return team_score, opponent_score

# Step 5: Model evaluation with visualization
def evaluate_model_with_viz(model, X_test_scaled, y_team_test, y_opponent_test):
    print("Generating prediction visualizations...")
    # Make predictions on test data
    predictions = model.predict(X_test_scaled)
    team_preds = predictions[0].flatten()
    opponent_preds = predictions[1].flatten()
    
    # Create a figure for visualization
    plt.figure(figsize=(15, 10))
    
    # Scatter plot of actual vs predicted team scores
    plt.subplot(2, 2, 1)
    plt.scatter(y_team_test, team_preds, alpha=0.5)
    plt.plot([0, max(y_team_test)], [0, max(y_team_test)], 'r--')
    plt.xlabel('Actual Team Score')
    plt.ylabel('Predicted Team Score')
    plt.title('Team Score: Actual vs Predicted')
    
    # Scatter plot of actual vs predicted opponent scores
    plt.subplot(2, 2, 2)
    plt.scatter(y_opponent_test, opponent_preds, alpha=0.5)
    plt.plot([0, max(y_opponent_test)], [0, max(y_opponent_test)], 'r--')
    plt.xlabel('Actual Opponent Score')
    plt.ylabel('Predicted Opponent Score')
    plt.title('Opponent Score: Actual vs Predicted')
    
    # Histogram of residuals for team scores
    plt.subplot(2, 2, 3)
    residuals_team = y_team_test - team_preds
    plt.hist(residuals_team, bins=20)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residuals for Team Score')
    
    # Histogram of residuals for opponent scores
    plt.subplot(2, 2, 4)
    residuals_opponent = y_opponent_test - opponent_preds
    plt.hist(residuals_opponent, bins=20)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residuals for Opponent Score')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    plt.show()
    
    # Calculate and print evaluation metrics
    team_mae = np.mean(np.abs(residuals_team))
    opponent_mae = np.mean(np.abs(residuals_opponent))
    team_mse = np.mean(residuals_team ** 2)
    opponent_mse = np.mean(residuals_opponent ** 2)
    
    print(f"Team Score - MAE: {team_mae:.4f}, MSE: {team_mse:.4f}")
    print(f"Opponent Score - MAE: {opponent_mae:.4f}, MSE: {opponent_mse:.4f}")
    
    # Calculate accuracy for predicting win/loss/tie
    actual_result = np.where(y_team_test > y_opponent_test, 'W', 
                             np.where(y_team_test < y_opponent_test, 'L', 'T'))
    pred_result = np.where(team_preds > opponent_preds, 'W', 
                           np.where(team_preds < opponent_preds, 'L', 'T'))
    result_accuracy = np.mean(actual_result == pred_result)
    
    print(f"Win/Loss/Tie Prediction Accuracy: {result_accuracy:.4f}")
    
    return team_mae, opponent_mae, result_accuracy

# Main execution flow
def main():
    # File path to your NHL data
    excel_file_path = 'nhl_gamelogs.xlsx'
    
    try:
        # Load and preprocess the data
        print("Starting data loading and preprocessing...")
        df = load_and_preprocess_data(excel_file_path)
        
        # Check if we have enough data after preprocessing
        if len(df) < 100:
            print(f"Warning: Dataset has only {len(df)} rows after preprocessing. Model may not be accurate.")
        
        # Engineer features
        print("Starting feature engineering...")
        X, y_team, y_opponent, encoder, features = engineer_features(df)
        
        # Build and train the model
        print("Starting model training...")
        model, scaler, history = build_and_train_model(X, y_team, y_opponent)
        
        # Split data for evaluation
        _, X_test, _, y_team_test, _, y_opponent_test = train_test_split(
            X, y_team, y_opponent, test_size=0.2, random_state=42
        )
        X_test_scaled = scaler.transform(X_test)
        
        # Evaluate model with visualizations
        evaluate_model_with_viz(model, X_test_scaled, y_team_test, y_opponent_test)
        
        # Example prediction for a future game
        print("\nGenerating example prediction...")
        team = df['Team'].iloc[0]  # Use first team from data as example
        
        # Create example stats based on averages from the data
        example_stats = {}
        for feature in features:
            if feature in df.columns and feature != 'Days_Since_Season_Start':
                example_stats[feature] = df[feature].mean()
        
        # Set days since season start to an appropriate value
        days_since_season_start = 180  # Example: mid-season
        
        # Make prediction
        team_score, opponent_score = predict_game_score(
            model, scaler, encoder, features, team, example_stats, days_since_season_start
        )
        
        print(f"\nPredicted score for {team}: {team_score}-{opponent_score}")
        
        # Save the model and related components
        print("Saving model and components...")
        model.save('nhl_score_prediction_model.h5')
        
        # Save feature names and encoder for future use
        import pickle
        with open('model_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('model_encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)
        with open('model_features.pkl', 'wb') as f:
            pickle.dump(features, f)
            
        print("Model and components saved successfully.")
        print("\nTo make predictions with this model, use the predict_game_score function.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
