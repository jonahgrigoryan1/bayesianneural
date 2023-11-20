from sklearn.impute import KNNImputer                                                                           
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Create an instance of KNNImputer                                                                              
# Impute missing values
imputer = KNNImputer(n_neighbors=5)
train_data_imputed = imputer.fit_transform(train_data)
train_data_imputed = pd.DataFrame(train_data_imputed, columns=train_data.columns)

# Apply Isolation Forest for outlier detection
iso_forest = IsolationForest(contamination=0.01)
outliers = iso_forest.fit_predict(train_data_imputed)
train_data_imputed['outlier'] = outliers

# Filter inliers                                                                                                
train_data_inliers = train_data_imputed[train_data_imputed['outlier'] == 1].drop('outlier', axis=1)

# Scale features
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data_inliers)
train_data_scaled_df = pd.DataFrame(train_data_scaled, columns=train_data_inliers.columns)

# Feature Engineering: Create temporal features
# For example, rolling averages and differences                                                                 
window_sizes = [3, 5, 10]  # Example window sizes for rolling features                                          
for window in window_sizes:                                                                                     
   train_data_inliers[f'wap_rolling_mean_{window}'] = train_data_inliers['wap'].rolling(window=window).mean()
   train_data_inliers[f'wap_rolling_std_{window}'] = train_data_inliers['wap'].rolling(window=window).std()    
   train_data_inliers[f'wap_diff_{window}'] = train_data_inliers['wap'].diff(periods=window)                   
                                                                                                               
# Drop rows with NaN values created by rolling features                                                         
train_data_inliers.dropna(inplace=True)

from sklearn.preprocessing import MinMaxScaler

# Assuming train_data_inliers has the rolling features added and NaN values dropped                             
# Apply MinMaxScaler to scale features to a range suitable for VAE (e.g., [0, 1])                               
scaler = MinMaxScaler(feature_range=(0, 1))                                                                    
train_data_scaled = scaler.fit_transform(train_data_inliers)                                                    
train_data_scaled_df = pd.DataFrame(train_data_scaled, columns=train_data_inliers.columns)
