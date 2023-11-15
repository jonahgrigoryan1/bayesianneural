from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import pandas as pd
import numpy as np

# Create an instance of KNNImputer
imputer = KNNImputer(n_neighbors=5)

# Fit the imputer to the dataset and transform it
train_data_imputed = imputer.fit_transform(train_data)

# Convert the array back to a pandas DataFrame
train_data_imputed = pd.DataFrame(train_data_imputed, columns=train_data.columns)

# Feature Engineering: Incorporate domain knowledge to create new features
# Example: train_data_imputed['new_feature'] = train_data_imputed['feature1'] / train_data_imputed['feature2']

# Advanced feature selection
# Example: pca = PCA(n_components=0.95)
# train_data_reduced = pca.fit_transform(train_data_imputed)

# Ensure all features are on a similar scale
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data_imputed)

# Identify and handle outliers
outlier_detector = EllipticEnvelope(contamination=0.01)
outliers = outlier_detector.fit_predict(train_data_scaled)
train_data_scaled = train_data_scaled[outliers == 1, :]

# Convert the array back to a pandas DataFrame after scaling and outlier removal
train_data_processed = pd.DataFrame(train_data_scaled, columns=train_data.columns)

# Check for missing values again
missing_values_per_column_after_processing = train_data_processed.isnull().sum()
total_missing_values_after_processing = train_data_processed.isnull().sum().sum()

missing_values_per_column_after_processing, total_missing_values_after_processing
