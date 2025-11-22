import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------
# Load dataset
# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("income_evaluation.csv")
    df.columns = df.columns.str.strip()   # remove spaces
    return df

df = load_data()

st.title("Income Prediction App (Random Forest)")
st.write("Predict whether a person's income is **>=50K or <50K**.")

# -----------------------------------------
# Preprocessing pipeline
# -----------------------------------------
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns.drop("income")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# -----------------------------------------
# Train model
# -----------------------------------------
X = df.drop("income", axis=1)
y = df["income"]

model.fit(X, y)

# -----------------------------------------
# User Input Form
# -----------------------------------------

st.subheader("Enter Person Details")

user_data = {}

for col in numeric_features:
    user_data[col] = st.number_input(col, value=float(df[col].median()))

for col in categorical_features:
    user_data[col] = st.selectbox(col, df[col].unique())

# Convert user input to DataFrame
user_df = pd.DataFrame([user_data])

# -----------------------------------------
# Prediction
# -----------------------------------------
if st.button("Predict Income"):
    prediction = model.predict(user_df)
    st.success(f"Predicted Income: **{prediction[0]}**")
