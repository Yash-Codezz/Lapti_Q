# === Importing Libraries ==============================

import time

import numpy as np

import pandas as pd 

from feature_engineering import feature_engineer

from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

import joblib


# === Importing Dataset ==============================

data = pd.read_csv("data/Laptop Dataset.csv")

data.drop(columns=['Series', 'Display Size (Inches)', 'Display Resolution'], inplace=True)


# === Feature Engineering & Data Cleaning ==============================

data.drop_duplicates(keep='first', inplace=True)

data.dropna(subset=['Price (Rs)'], inplace=True)


# === Spliting Data ==============================

X = data.drop(columns=['Price (Rs)'])
y = data['Price (Rs)']


# === Preprocessing Pipeline ==============================

cat_pipeline = Pipeline(steps=[
    ('Impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('Encode', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.01))
])

display_pipeline = Pipeline(steps=[
    ('Impute Display', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('Encode Display', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.001))
])

preprocessing = ColumnTransformer(transformers=[
    ('Impute Median', SimpleImputer(strategy='median'), ['Weight (Kg)', 'Pixel Density (PPI)', 'Clock Speed (GHz)', 'RAM Speed (MHz)', 'Refresh Rate (Hz)']),
    ('Impute Constant', SimpleImputer(strategy='constant', fill_value=0), ['Graphics Memory (GB)', 'SSD Capacity (GB)', 'HDD Capacity (GB)']),
    ('Feature Engineer Categorical', cat_pipeline, ['Display Touchscreen', 'RAM Type', 'GPU Brand', 'GPU Series', 'Brand', 'Operating System', 'CPU Brand', 'CPU Segment', 'CPU Series']),
    ('Feature Engineer Display Type', display_pipeline, ['Display Type'])
], remainder='passthrough')


# == Building Model ============================

model = Pipeline(steps=[
    ('Feature Engineering', FunctionTransformer(feature_engineer)),
    ('Preprocessing', preprocessing),
    ('XGB', XGBRegressor(objective='reg:absoluteerror', learning_rate=0.1, max_depth=7, n_estimators=700, reg_alpha=10, reg_lambda=1, n_jobs=1))
])

start_time = time.time()
model.fit(X, y)
end_time = time.time() - start_time

print("Model Trained Successfully!")
print(f"Time Taken: {end_time:.2f}s")


# === Model Evaluation ==============================

cv = KFold(n_splits=10, shuffle=True, random_state=69)

print(f"\n{'='*30}\n")
print(f"r2-Score: {np.mean(cross_val_score(estimator=model, X=X, y=y, scoring='r2', cv=cv, n_jobs=-1)) * 100:.2f}%")
print(f"M.A.P.E: {-np.mean(cross_val_score(estimator=model, X=X, y=y, scoring='neg_mean_absolute_percentage_error', cv=cv, n_jobs=-1)) * 100:.2f}%")
print(f"\n{'='*30}\n")


# === Saving Model ==============================

joblib.dump(model, 'model/Lapti_Q.pkl')
print("Model Saved Successfully!")