import os
import ast
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "movies_metadata.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "movie_rating_model.pkl"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading dataset...")

    print(
        "Dataset path:",
        DATA_PATH
    )

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nDataset not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )

    print(
        "Dataset shape:",
        df.shape
    )

    print("\nColumns:")

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# EXTRACT FIRST GENRE
# ============================================================

def extract_genre(value):

    try:

        if pd.isna(value):

            return "Unknown"

        genres = ast.literal_eval(
            str(value)
        )

        if isinstance(
            genres,
            list
        ):

            names = [
                item.get(
                    "name",
                    ""
                )

                for item in genres

                if isinstance(
                    item,
                    dict
                )
            ]

            names = [
                name

                for name in names

                if name
            ]

            return (
                names[0]
                if names
                else "Unknown"
            )

    except Exception:

        pass

    return "Unknown"


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    columns = [
        "genres",
        "release_date",
        "runtime",
        "budget",
        "revenue",
        "vote_count",
        "vote_average"
    ]

    missing_columns = [
        column

        for column in columns

        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(missing_columns)
        )

    df = df[
        columns
    ].copy()

    # Extract genre

    df["Genre"] = df[
        "genres"
    ].apply(
        extract_genre
    )

    # Extract year

    df["Year"] = pd.to_datetime(
        df["release_date"],
        errors="coerce"
    ).dt.year

    # Convert numeric columns

    numeric_columns = [
        "runtime",
        "budget",
        "revenue",
        "vote_count",
        "vote_average"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Rename columns

    df = df.rename(
        columns={
            "runtime": "Runtime",
            "budget": "Budget",
            "revenue": "Gross",
            "vote_count": "Votes",
            "vote_average": "Rating"
        }
    )

    # Select required columns

    df = df[
        [
            "Genre",
            "Year",
            "Runtime",
            "Budget",
            "Gross",
            "Votes",
            "Rating"
        ]
    ]

    # Remove missing target

    df = df.dropna(
        subset=[
            "Rating"
        ]
    )

    # Keep valid rating range

    df = df[
        (df["Rating"] >= 0)
        &
        (df["Rating"] <= 10)
    ]

    # Keep movies with votes

    df = df[
        df["Votes"] > 0
    ]

    return df


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor():

    categorical_features = [
        "Genre"
    ]

    numerical_features = [
        "Year",
        "Runtime",
        "Budget",
        "Gross",
        "Votes"
    ]

    # Numerical preprocessing

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combine preprocessing

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test
):


    models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=80,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}

        
    results = {}

    best_model = None

    best_model_name = None

    best_rmse = float(
        "inf"
    )

    # Train each model

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        # Create fresh preprocessor

        preprocessor = (
            build_preprocessor()
        )

        # Create pipeline

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )

        # Train

        pipeline.fit(
            X_train,
            y_train
        )

        # Predict

        predictions = pipeline.predict(
            X_test
        )

        # MAE

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        # MSE

        mse = mean_squared_error(
            y_test,
            predictions
        )

        # RMSE

        rmse = mse ** 0.5

        # R2

        r2 = r2_score(
            y_test,
            predictions
        )

        # Store results

        results[name] = {

            "MAE":
                float(mae),

            "RMSE":
                float(rmse),

            "R2":
                float(r2)
        }

        print(
            f"MAE  : {mae:.4f}"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

        print(
            f"R2   : {r2:.4f}"
        )

        # Select best model
        # based on lowest RMSE

        if rmse < best_rmse:

            best_rmse = rmse

            best_model = pipeline

            best_model_name = name

    return (
        best_model,
        best_model_name,
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "AI MOVIE RATING PREDICTOR"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    print(
        "\nCleaning dataset..."
    )

    df = clean_data(
        df
    )

    print(
        "\nCleaned dataset shape:",
        df.shape
    )

    print(
        "\nProcessed columns:"
    )

    print(
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # FEATURES AND TARGET
    # --------------------------------------------------------

    X = df[
        [
            "Genre",
            "Year",
            "Runtime",
            "Budget",
            "Gross",
            "Votes"
        ]
    ]

    y = df[
        "Rating"
    ]

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42
    )

    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    (
        best_model,
        best_model_name,
        results
    ) = train_models(

        X_train,

        X_test,

        y_train,

        y_test
    )

    # --------------------------------------------------------
    # CREATE MODEL DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {

        "model_name":
            best_model_name,

        "target":
            "Rating",

        "features": [

            "Genre",

            "Year",

            "Runtime",

            "Budget",

            "Gross",

            "Votes"
        ],

        "results":
            results
    }

    joblib.dump(
        metadata,
        METADATA_PATH
    )

    # --------------------------------------------------------
    # BEST MODEL METRICS
    # --------------------------------------------------------

    best_mae = results[
        best_model_name
    ][
        "MAE"
    ]

    best_rmse = results[
        best_model_name
    ][
        "RMSE"
    ]

    best_r2 = results[
        best_model_name
    ][
        "R2"
    ]

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 60
    )

    for name, metrics in results.items():

        print(
            f"\n{name}"
        )

        print(
            f"MAE  : {metrics['MAE']:.4f}"
        )

        print(
            f"RMSE : {metrics['RMSE']:.4f}"
        )

        print(
            f"R2   : {metrics['R2']:.4f}"
        )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "BEST MODEL"
    )

    print(
        "=" * 60
    )

    print(
        "Model:",
        best_model_name
    )

    print(
        f"MAE  : {best_mae:.4f}"
    )

    print(
        f"RMSE : {best_rmse:.4f}"
    )

    print(
        f"R2   : {best_r2:.4f}"
    )

    # --------------------------------------------------------
    # SAVED FILES
    # --------------------------------------------------------

    print(
        "\nModel saved:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nMetadata saved:"
    )

    print(
        METADATA_PATH
    )

    print(
        "\nTraining completed successfully!"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()