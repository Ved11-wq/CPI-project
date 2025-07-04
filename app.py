from flask import Flask, render_template, request, jsonify, json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import os
import json

app = Flask(__name__)

# Load model, scaler, and expected column order
model = joblib.load('Training/cpi_model.pkl')
scaler = joblib.load('Training/scaler.pkl')
feature_columns = joblib.load('Training/feature_columns.pkl')

# Load numerical columns from training data
numerical_cols_path = 'Training/numerical_columns.pkl'
if os.path.exists(numerical_cols_path):
    numerical_cols = joblib.load(numerical_cols_path)
else:
    # Fallback: Get numerical columns from feature_columns
    numerical_cols = [col for col in feature_columns if col not in ['Sector', 'Month'] and not col.startswith('Housing_')]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Get form inputs
        sector = request.form.get('sector', 'Rural')
        month = request.form.get('month', 'January')
        year = float(request.form.get('year', 2023))
        housing_value = float(request.form.get('housing_value', 100))

        # Debug: Print all form data
        print("\n=== Form Data ===")
        for key, value in request.form.items():
            print(f"{key}: {value}")

        # Base numeric values for features
        base_value = 100

        # Load feature columns from training data
        feature_columns = joblib.load('Training/feature_columns.pkl')
        numerical_cols = joblib.load('Training/numerical_columns.pkl')
        
        # Debug: Print loaded columns
        print("\n=== Feature Columns ===")
        print(f"Feature columns: {feature_columns}")
        print(f"Numerical columns: {numerical_cols}")
        
        # Create a DataFrame with all features initialized to base value
        input_df = pd.DataFrame({
            feature: [base_value] for feature in feature_columns
        })
        input_df['Year'] = year
        
        # Remove target column if present
        if 'General index' in input_df.columns:
            input_df = input_df.drop(['General index'], axis=1)
            
        # Set numeric features from form
        for feature in numerical_cols:
            try:
                val = float(request.form.get(feature, base_value))
                # Ensure values are within reasonable range (0-200)
                if val < 0:
                    val = base_value
                elif val > 200:
                    val = 200
                input_df[feature] = val
            except (ValueError, TypeError) as e:
                print(f"Error setting {feature}: {e}")
                input_df[feature] = base_value
        
        # Debug: Print input DataFrame before one-hot encoding
        print("\n=== Input DataFrame (before one-hot) ===")
        print(input_df.head())

        # Set categorical features
        print("\n=== Setting Categorical Features ===")
        print(f"Sector: {sector}, Month: {month}, Housing Value: {housing_value}")
        
        # Reset all categorical columns to 0 first
        for col in input_df.columns:
            if col.startswith(('Sector_', 'Month_', 'Housing_')):
                input_df[col] = 0
        
        # Set sector
        if sector == 'Urban':
            if 'Sector_Urban' in input_df.columns:
                input_df['Sector_Urban'] = 1
                print("Set Sector_Urban to 1")
            else:
                print("Warning: Sector_Urban column not found")
        elif sector == 'Rural+Urban':
            if 'Sector_Rural+Urban' in input_df.columns:
                input_df['Sector_Rural+Urban'] = 1
                print("Set Sector_Rural+Urban to 1")
            else:
                print("Warning: Sector_Rural+Urban column not found")
        
        # Set month
        month_col = f'Month_{month}'
        if month != 'January' and month_col in input_df.columns:
            input_df[month_col] = 1
            print(f"Set {month_col} to 1")
        
        # Set housing value
        housing_col = f'Housing_{housing_value}'
        if housing_col in input_df.columns:
            input_df[housing_col] = 1
            print(f"Set {housing_col} to 1")
        
        # Debug: Print final input DataFrame
        print("\n=== Final Input DataFrame ===")
        print("Columns:", input_df.columns.tolist())
        print("Shape:", input_df.shape)
        print("First few rows:")
        print(input_df.head())

        # Scale features
        try:
            # Get the columns used during training (excluding target)
            training_columns = [col for col in feature_columns if col != 'General index']
            
            # Ensure we have all columns expected by the scaler
            missing_cols = set(training_columns) - set(input_df.columns)
            if missing_cols:
                print(f"\n=== Missing Columns ===")
                print(f"Adding {len(missing_cols)} missing columns with 0 values")
                for col in missing_cols:
                    input_df[col] = 0
            
            # Reorder columns to match training order
            input_df = input_df[training_columns]
            
            # Debug: Verify final columns before scaling
            print("\n=== Columns Before Scaling ===")
            print(f"Expected columns: {training_columns}")
            print(f"Actual columns: {input_df.columns.tolist()}")
            
            scaled_input = scaler.transform(input_df)
            print("\n=== Scaled Input ===")
            print(f"Shape: {scaled_input.shape}")
            print("First 10 values:", scaled_input[0][:10])
            
            # Make prediction
            prediction = model.predict(scaled_input)[0]
            print(f"\n=== Prediction ===")
            print(f"Raw prediction: {prediction}")
            
            # Clamp prediction to reasonable range
            if prediction < 0:
                print("Warning: Negative prediction, clamping to 0")
                prediction = 0
            elif prediction > 200:
                print("Warning: Prediction > 200, clamping to 200")
                prediction = 200
            
            # Prepare debug info
            debug_info = {
                'prediction': prediction,
                'form_data': dict(request.form),
                'feature_columns': feature_columns,
                'input_shape': input_df.shape,
                'scaled_input_shape': scaled_input.shape,
                'model_type': str(type(model))
            }
            
            # Render the result template
            return render_template('result.html', 
                                 prediction=prediction,
                                 debug_info=debug_info,
                                 debug=True)  # Set to False in production
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            error_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Error</title>
                <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
            </head>
            <body>
                <div class="navbar">
                    <a href="{{ url_for('home') }}">Home</a>
                    <a href="{{ url_for('predict') }}">Predict CPI</a>
                </div>

                <div class="container">
                    <h2>Error</h2>
                    <div class="error-message">
                        <p>An error occurred during prediction: {str(e)}</p>
                        <a href="{{ url_for('predict') }}" class="back-button">Try Again</a>
                    </div>
                </div>

                <footer>
                    <p>&copy; 2025 CPI Prediction App</p>
                </footer>
            </body>
            </html>
            """
            return error_html

    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

if __name__ == '__main__':
    app.run(debug=True)
