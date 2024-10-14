import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset harga rumah
data = pd.read_csv("housing.csv")

# Pisahkan fitur dan target
X = data[["total_bedrooms", "total_rooms", "median_income", "ocean_proximity"]]  # Fitur
y = data["median_house_value"]  # Target

# Buat pipeline untuk preprocessing
numerical_features = ["total_bedrooms", "total_rooms", "median_income"]
categorical_features = ["ocean_proximity"]

# Buat transformer untuk fitur numerik dan kategorik
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Gabungkan preprocessing menjadi satu pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Buat pipeline lengkap dengan model
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model_pipeline.fit(X_train, y_train)

def predict(total_bedrooms, total_rooms, median_income, ocean_proximity):
    # Buat DataFrame untuk input baru
    input_data = pd.DataFrame({
        "total_bedrooms": [total_bedrooms],
        "total_rooms": [total_rooms],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })
    # Lakukan prediksi menggunakan pipeline
    return model_pipeline.predict(input_data)[0]

def get_statistics():
    return {
        "mean": np.mean(y),
        "median": np.median(y),
        "std_dev": np.std(y)
    }
