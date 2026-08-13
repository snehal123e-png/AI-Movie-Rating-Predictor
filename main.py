import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "movie_rating_model.pkl"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Movie Rating Predictor API",
    description="Machine Learning API for predicting IMDb movie ratings",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

model = None
model_error = None

try:

    print("=" * 60)
    print("LOADING AI MOVIE RATING MODEL")
    print("=" * 60)

    print("Model path:")
    print(MODEL_PATH)

    print("Model exists:")
    print(os.path.exists(MODEL_PATH))

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")

except Exception as e:

    model = None
    model_error = str(e)

    print("MODEL LOADING FAILED!")
    print("Error:", e)


# ============================================================
# INPUT MODEL
# ============================================================

class MovieInput(BaseModel):

    genre: str = Field(
        ...,
        min_length=1
    )

    runtime: float = Field(
        ...,
        gt=0
    )

    budget: float = Field(
        ...,
        ge=0
    )

    year: int = Field(
        ...,
        ge=1900,
        le=2100
    )

    votes: float = Field(
        ...,
        ge=0
    )

    gross: float = Field(
        ...,
        ge=0
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "success",
        "message": "AI Movie Rating Predictor API is running"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "error": model_error
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(movie: MovieInput):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="ML model is not loaded"
        )

    input_data = pd.DataFrame(
        [
            {
                "Genre": movie.genre,
                "Year": movie.year,
                "Runtime": movie.runtime,
                "Budget": movie.budget,
                "Gross": movie.gross,
                "Votes": movie.votes
            }
        ]
    )

    try:

        prediction = model.predict(
            input_data
        )[0]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    prediction = max(
        0,
        min(
            10,
            float(prediction)
        )
    )

    if prediction >= 8:

        category = "Excellent"

    elif prediction >= 7:

        category = "Very Good"

    elif prediction >= 6:

        category = "Good"

    elif prediction >= 5:

        category = "Average"

    else:

        category = "Below Average"

    return {
        "predicted_rating": round(
            prediction,
            2
        ),
        "category": category,
        "message": "Rating predicted successfully"
    }