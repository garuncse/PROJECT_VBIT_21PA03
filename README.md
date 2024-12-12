# PROJECT_VBIT_21PA03
# Data analytics meta- model for iot using hybrid machine learning techniques 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load CSV data
file_path = "/content/pcos_fluctuating_data.csv"  # Replace with the correct file path
df = pd.read_csv(file_path)

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Convert 'timestamp' to datetime and sort data
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])  # Drop rows with invalid timestamps
df = df.sort_values(by=['patient_id', 'timestamp'])

# Define acceptable ranges with lowercase keys
ranges = {
    "lh": (1.0, 10.0),
    "fsh": (1.5, 12.0),
    "testosterone": (20.0, 80.0),
    "insulin": (5.0, 25.0),
    "cycleregularity": (20.0, 35.0),
    "ovariansize": (3.0, 8.0),
    "folliclescount": (10.0, 15.0),
    "endometrialthickness": (5.0, 15.0),
    "cysts": (0.0, 5.0),
}

# Fuzzy Logic Risk Levels
def fuzzy_risk(row):
    fuzzy_score = 0
    for col, (low, high) in ranges.items():
        if row[col] < low:
            fuzzy_score += (low - row[col]) / low  # Penalize low values
        elif row[col] > high:
            fuzzy_score += (row[col] - high) / high  # Penalize high values
    return fuzzy_score / len(ranges)  # Normalize to average risk

df['fuzzy_risk'] = df.apply(fuzzy_risk, axis=1)

# Risk levels and advice
def risk_assessment(row):
    risk_status = {}
    for col, (low, high) in ranges.items():
        if row[col] < low:
            risk_status[col] = ("Low", f"Increase {col} levels with {col}-enhancing therapies.")
        elif row[col] > high:
            risk_status[col] = ("High", f"Reduce {col} levels with {col}-reducing therapies.")
        else:
            risk_status[col] = ("Normal", "No immediate action required.")
    return risk_status

df['risk_status'] = df.apply(risk_assessment, axis=1)

# Adaptive Moving Window Regression (AMWR) for Trend Detection
def amwr_trend(data, feature):
    trends = []
    window_size = 5  # Define window size for trend detection
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        trend = np.polyfit(range(window_size), window[feature], 1)[0]  # Linear regression slope (trend)
        trends.append(trend)
    return trends + [None] * (window_size - 1)  # Padding None for the first few rows

df['trend_lh'] = amwr_trend(df, 'lh')

# LSTM for Prediction of Future Hormonal Levels
def train_lstm(patient_data, feature, n_steps=3):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(patient_data[[feature]])

    X, y = [], []
    for i in range(len(data_scaled) - n_steps):
        X.append(data_scaled[i:i+n_steps, 0])
        y.append(data_scaled[i+n_steps, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=0, batch_size=16)

    # Predictions
    y_pred = model.predict(X, verbose=0)
    y_pred = scaler.inverse_transform(y_pred)
    return model, y_pred.flatten()

# Federated Learning Simulation (model aggregation)
def federated_averaging(models):
    """Perform Federated Averaging to combine model weights."""
    weights = [model.get_weights() for model in models]  # Get model weights
    avg_weights = [np.mean([w[i] for w in weights], axis=0) for i in range(len(weights[0]))]

    # Create a new model and set the aggregated weights
    new_model = Sequential([
        LSTM(50, activation='relu', input_shape=(3, 1)),
        Dense(1)
    ])
    new_model.set_weights(avg_weights)  # Set averaged weights to the new model

    return new_model

# Federated Learning Process
models = []
patients = df['patient_id'].unique()

# Local training on each patient's data
for patient_id in patients:
    patient_data = df[df['patient_id'] == patient_id].reset_index(drop=True)
    model, _ = train_lstm(patient_data, feature='lh')  # Train model on local data
    models.append(model)

# Perform Federated Averaging to aggregate model weights
global_model = federated_averaging(models)

# Generate Report for Each Patient
for patient_id in patients:
    patient_data = df[df['patient_id'] == patient_id].reset_index(drop=True)
    recent_data = patient_data.tail(5)

    # Risk Levels Table
    risk_table = PrettyTable()
    risk_table.field_names = ["Timestamp"] + list(ranges.keys()) + ["Fuzzy Risk"]
    for _, row in recent_data.iterrows():
        risk_table.add_row(
            [row['timestamp']] +
            [f"{row[col]:.2f}" for col in ranges.keys()] +
            [f"{row['fuzzy_risk']:.2f}"]
        )
    print(f"\nRisk Levels for Patient {patient_id}:\n{risk_table}")

    # Advice Table
    advice_table = PrettyTable()
    advice_table.field_names = ["Metric", "Status", "Advice"]
    for _, row in recent_data.iterrows():
        for metric, (status, advice) in row['risk_status'].items():
            if status != "Normal":
                advice_table.add_row([metric, status, advice])
    print(f"\nTreatment and Advice for Patient {patient_id}:\n{advice_table}")

    # Prediction and Summary
    _, predictions = train_lstm(patient_data, feature='lh')
    avg_risk = recent_data['fuzzy_risk'].mean()
    lh_trend = recent_data['trend_lh'].dropna().mean()

    summary_table = PrettyTable()
    summary_table.field_names = ["Patient ID", "Average Risk", "LH Trend", "Recommendation"]
    summary_table.add_row([patient_id, f"{avg_risk:.2f}",
                          "Increasing" if lh_trend > 0 else "Decreasing" if lh_trend < 0 else "Stable",
                          "Stabilize LH levels to avoid invasive procedures."])
    print(f"\nSummary for Patient {patient_id}:\n{summary_table}")

    # Plot Health Risks and Balancing Strategies
    plt.figure(figsize=(12, 6))

    # Plot Fuzzy Risk vs LH Levels
    plt.subplot(1, 2, 1)
    plt.plot(recent_data['timestamp'], recent_data['lh'], label='LH Levels', marker='o')
    plt.plot(recent_data['timestamp'], recent_data['fuzzy_risk'], label='Fuzzy Risk', linestyle='--', marker='x')
    plt.title(f"Health Risks and LH Levels for Patient {patient_id}")
    plt.xlabel('Timestamp')
    plt.ylabel('Level / Risk')
    plt.legend()

    # Plot LH Trend vs Fuzzy Risk
    plt.subplot(1, 2, 2)
    plt.plot(recent_data['timestamp'], recent_data['trend_lh'], label='LH Trend', color='orange', marker='o')
    plt.plot(recent_data['timestamp'], recent_data['fuzzy_risk'], label='Fuzzy Risk', color='green', linestyle='--', marker='x')
    plt.title(f"LH Trend and Fuzzy Risk for Patient {patient_id}")
    plt.xlabel('Timestamp')
    plt.ylabel('Trend / Risk')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

