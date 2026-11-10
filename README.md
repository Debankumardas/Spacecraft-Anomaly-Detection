# AI-Powered Spacecraft Anomaly Detection

An AI-powered time-series anomaly detection system designed to identify unusual patterns in spacecraft telemetry using NASA's SMAP and MSL datasets.

## Project Overview

Spacecraft and planetary rovers generate telemetry data through onboard sensors and systems. Detecting unusual patterns in this data can help identify potential operational anomalies.

This project explores statistical and deep learning approaches to spacecraft telemetry anomaly detection.

## Objectives

- Explore and analyse NASA SMAP and MSL telemetry datasets.
- Develop a reliable time-series preprocessing pipeline.
- Establish a statistical anomaly detection baseline.
- Build an LSTM-based forecasting model.
- Calculate prediction errors and anomaly scores.
- Implement dynamic anomaly thresholding.
- Evaluate models using point-level and event-level metrics.
- Develop an interactive Streamlit dashboard.

## Technology Stack

- Python
- NumPy and Pandas
- Scikit-learn
- PyTorch
- Matplotlib and Plotly
- Streamlit
- Git and GitHub

## Project Structure

```text
app/            Streamlit application
configs/        Project configuration
data/           Dataset storage
notebooks/      Data exploration and experiments
src/            Core project modules
models/         Trained model artifacts
reports/        Results and visualisations
tests/          Automated tests
```

## Dataset

The project is designed to use NASA SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory) telemetry anomaly benchmark data.

Dataset source and usage details will be documented after dataset verification.

## Project Status

Currently in the initial project setup and architecture stage.

## Disclaimer

This is an educational and research-oriented anomaly detection project. It is not a certified spacecraft monitoring or operational safety system.
