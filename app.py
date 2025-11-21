#
# Flask Backend API: app.py (FIXED BINDING ERROR)
#
# This API provides endpoints for:
# 1. Prediction of exam scores using the trained ML model.
# 2. Managing student tasks (Add, Get, Update, Delete).
#
# USAGE: Run 'python app.py'
# NOTE: Ensure rf_productivity_model.pkl, feature_scaler.pkl, and feature_names.pkl
# are in the same directory.
#

import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
# Enable CORS for the React Native app running on Expo
CORS(app)

# --- Configuration ---
DATABASE_NAME = 'study_plan.db'
MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) # Current directory

# --- Model Loading ---
try:
    with open(os.path.join(MODEL_DIR, 'rf_productivity_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f:
        FEATURE_NAMES = pickle.load(f)

    print("ML Model, Scaler, and Feature Names loaded successfully.")
    print("Expected Model Features:", FEATURE_NAMES)
except FileNotFoundError as e:
    print(f"ERROR: Model/Scaler file not found. Have you run the Colab notebook and downloaded the files? ({e})")
    model = None
    scaler = None
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    model = None
    scaler = None

# --- Database Setup (SQLite) ---
def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    # Table to store tasks
    c.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            is_completed INTEGER NOT NULL DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

# Initialize the database when the app starts
init_db()

# --- Utility Function for Preprocessing Prediction Data ---
def preprocess_input(input_data):
    """
    Transforms raw input data into the format expected by the ML model.
    """
    try:
        # Create a single row DataFrame based on model's expected features
        
        # 1. Base DataFrame from the required FEATURE_NAMES (excluding dummy variables)
        base_features = [
            'study_hours_per_day', 'social_media_hours', 'netflix_hours',
            'part_time_job/tuition', 'attendance_percentage', 'sleep_hours',
            'previous_gpa', 'family_income_range', 'time_management_score'
        ]
        
        data_row = {
            f: [input_data.get(f)] for f in base_features
        }
        
        # Add dummy variables for internet_quality
        # Ensure that 'internet_q_Low' and 'internet_q_Medium' are accounted for, 
        # as 'internet_q_High' is the reference category (0/0 in the dummies).
        
        for feature in FEATURE_NAMES:
            if feature.startswith('internet_q_'):
                data_row[feature] = [0]
        
        internet_quality = input_data.get('internet_quality', 'High') # Default to High
        
        if internet_quality == 'Low' and 'internet_q_Low' in FEATURE_NAMES:
            data_row['internet_q_Low'] = [1]
        elif internet_quality == 'Medium' and 'internet_q_Medium' in FEATURE_NAMES:
            data_row['internet_q_Medium'] = [1]
        
        df_input = pd.DataFrame(data_row)
        
        # 2. Apply categorical/ordinal mappings used in Colab
        
        # Family Income Range (Low=1, Medium=2, High=3)
        income_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        df_input['family_income_range'] = df_input['family_income_range'].replace(income_mapping)

        # part_time_job/tuition (Yes=1, No=0)
        # Handle both string input (from frontend) and potentially numeric (from previous state)
        df_input['part_time_job/tuition'] = df_input['part_time_job/tuition'].apply(lambda x: 1 if x == 1 or x == 'Yes' else 0)

        # 3. Align columns to the exact order and names expected by the model
        df_aligned = df_input.reindex(columns=FEATURE_NAMES, fill_value=0)

        # 4. Scale the aligned features
        scaled_data = scaler.transform(df_aligned)
        
        return scaled_data

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


# --- API Endpoints ---

@app.route('/', methods=['GET'])
def health_check():
    """Simple health check and instructions for base URL access."""
    return jsonify({
        "status": "API running successfully",
        "message": "Use /predict for ML predictions or /tasks for task management.",
        "model_status": "Loaded" if model else "Failed to load"
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for predicting exam score based on student habits."""
    if not model or not scaler:
        return jsonify({'error': 'Prediction model is not loaded.'}), 500

    data = request.json
    if not data:
        return jsonify({'error': 'No input data provided.'}), 400
    
    # Check if all required fields are present and valid
    required_keys = ['study_hours_per_day', 'social_media_hours', 'netflix_hours',
                     'part_time_job_tuition', 'attendance_percentage', 'sleep_hours',
                     'internet_quality', 'previous_gpa', 'family_income_range',
                     'time_management_score']
    
    for key in required_keys:
        # Note: The frontend sends 'part_time_job_tuition', not 'part_time_job/tuition'
        if key not in data and key != 'part_time_job/tuition': 
            return jsonify({'error': f'Missing required input field: {key}'}), 400
    
    # Preprocess the input data
    processed_data = preprocess_input(data)

    if processed_data is not None:
        # Perform prediction
        prediction = model.predict(processed_data)[0]
        
        # Format the prediction to be a nice integer or two decimal points
        predicted_score = round(max(0, min(100, prediction))) # Ensure score is between 0 and 100
        
        return jsonify({
            'predicted_score': predicted_score,
            'message': 'Prediction successful.'
        })
    else:
        return jsonify({'error': 'Failed to process input data.'}), 400


# --- Task Management Endpoints ---

@app.route('/tasks', methods=['POST'])
def add_task():
    """Endpoint to add a new task."""
    data = request.json
    required = ['title', 'date', 'time']
    if not all(k in data for k in required):
        return jsonify({'error': 'Missing task fields (title, date, time)'}), 400

    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO tasks (title, date, time, is_completed) VALUES (?, ?, ?, 0)",
                  (data['title'], data['date'], data['time']))
        conn.commit()
        task_id = c.lastrowid
        conn.close()
        return jsonify({'message': 'Task added successfully', 'id': task_id, **data}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tasks', methods=['GET'])
def get_tasks():
    """Endpoint to retrieve all incomplete tasks, ordered by date and time."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.row_factory = sqlite3.Row # Allows access by column name
        c = conn.cursor()
        # Order by date (ISO format 'YYYY-MM-DD') and time ('HH:MM')
        c.execute("SELECT * FROM tasks WHERE is_completed = 0 ORDER BY date, time")
        tasks = [dict(row) for row in c.fetchall()]
        conn.close()
        return jsonify(tasks), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    """Endpoint to edit or complete a task."""
    data = request.json
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    
    try:
        # Handle marking as complete (which automatically removes it as per requirement)
        if 'is_completed' in data and data['is_completed'] == 1:
            c.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()
            conn.close()
            return jsonify({'message': f'Task {task_id} marked as complete and removed.'}), 200
            
        # Handle general update (edit)
        updates = []
        params = []
        if 'title' in data:
            updates.append("title = ?")
            params.append(data['title'])
        if 'date' in data:
            updates.append("date = ?")
            params.append(data['date'])
        if 'time' in data:
            updates.append("time = ?")
            params.append(data['time'])

        if updates:
            query = "UPDATE tasks SET " + ", ".join(updates) + " WHERE id = ?"
            params.append(task_id)
            c.execute(query, tuple(params))
            conn.commit()
            conn.close()
            if c.rowcount == 0:
                return jsonify({'error': 'Task not found'}), 404
            return jsonify({'message': f'Task {task_id} updated successfully'}), 200
        else:
            conn.close()
            return jsonify({'error': 'No fields provided for update'}), 400
            
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500


@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    """Endpoint to delete a task."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        
        if c.rowcount == 0:
            conn.close()
            return jsonify({'error': 'Task not found'}), 404
            
        conn.close()
        return jsonify({'message': f'Task {task_id} deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Running the App (FIXED) ---
if __name__ == '__main__':
    # BINDING FIX: Use '0.0.0.0' to listen on all interfaces.
    # This allows external devices (like your phone) to connect regardless of the 
    # specific IP address your MacBook currently holds (e.g., 192.168.0.x).
    print("Attempting to run Flask on 0.0.0.0:5000. Use your MacBook's current Wi-Fi IP in the React Native app.")
    app.run(host='0.0.0.0', port=5000, debug=True)