# SymptoRx: AI-Powered Health Insights API

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

Haven Health provides a robust, scalable RESTful API for symptom-based disease prediction and medication recommendations. It leverages a trained machine learning model to deliver real-time insights from a trusted medical knowledge base.

## Table of Contents

- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)


## Key Features

*   **Symptom-Based Disease Prediction**: Utilizes a Random Forest classifier to predict likely diseases from user-provided symptoms with over 90% accuracy.
*   **Medicine Recommendation**: Retrieves relevant medication suggestions from a structured dataset based on the predicted disease.
*   **High-Performance API**: Built with FastAPI for asynchronous, high-speed request handling, serving predictions with sub-second latency.
*   **Automatic API Documentation**: Interactive API documentation (Swagger UI and ReDoc) is automatically generated for easy testing and integration.

## Tech Stack

| Category           | Technology                                        |
| ------------------ | ------------------------------------------------- |
| **Backend**        | Python, FastAPI, Uvicorn, Pydantic                |
| **Machine Learning** | scikit-learn (Random Forest), Pandas, NumPy, Pickle |
| **Frontend**       | Jinja2, HTML, CSS (for demonstration purposes)    |

## System Architecture

The application is built on a service-oriented architecture. The **FastAPI** backend serves as the core, exposing an endpoint for disease prediction.

1.  **Prediction Service**: When a user sends a list of symptoms to the `/predict` endpoint, the service vectorizes the input and feeds it into the pre-trained **scikit-learn** model.
2.  **Medicine Lookup Service**: The `/medicines/{disease}` endpoint allows direct querying for medications by disease name, retrieving data from a CSV dataset using **Pandas**.

## Getting Started

Follow these instructions to get a local copy up and running for development and testing purposes.

### Prerequisites

*   Python 3.10 or higher
*   Git
*   A virtual environment tool (e.g., `venv`, `conda`)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Gaurabh007/SymptoRx-AI.git
    cd SymptoRx-AI
    ```

2.  **Create and activate a virtual environment:**
    *   On macOS/Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### Configuration

Create a `.env` file in the root of the project to manage file paths.

```env
# File Paths
MODEL_PATH="models/random_forest.pkl"
SYMPTOM_DATA_PATH="data/training.csv"
MEDICINE_DATA_PATH="data/medicines.csv"
```

## Usage

Once the installation and configuration are complete, run the FastAPI application using Uvicorn:

```sh
uvicorn app.main:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

*   **Interactive API Docs (Swagger)**: `http://127.0.0.1:8000/docs`
*   **Alternative API Docs (ReDoc)**: `http://127.0.0.1:8000/redoc`

## API Endpoints

### Predict Disease

*   **Endpoint**: `POST /predict`
*   **Description**: Predicts a disease based on a list of symptoms and returns relevant medication suggestions.
*   **Request Body**:
    ```json
    {
      "symptoms": ["fever", "cough", "headache"]
    }
    ```
*   **Success Response** (`200 OK`):
    ```json
    {
      "disease": "Influenza",
      "confidence_score": 0.92,
      "recommended_medications": ["Oseltamivir", "Ibuprofen"]
    }
    ```

## Project Structure

```
.
├── data/
│   ├── disease_prediction_dataset/
│   │   └── Training.csv
│   └── med_dataset/
│       └── disease2med.csv
├── Frontend/
│   ├── images/
│   ├── drop_down.css
│   ├── drop_down.html
│   ├── drop_down.js
│   └── styles.css
├── models/
│   └── randomforest.pkl
├── research/
│   ├── medication.ipynb
│   ├── medicine-recommendation-system.ipynb
│   └── model_training.ipynb
├── app.py
├── README.md
└── requirements.txt

```


