import pandas as pd
from sklearn.impute import KNNImputer                                                                           
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

train_data = pd.read_csv("subset_train.csv")

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
train_data_inliers.reset_index(drop=True, inplace=True)

from sklearn.preprocessing import MinMaxScaler

original_data = pd.read_csv('subset_train.csv')

# Assuming train_data_inliers has the rolling features added and NaN values dropped                             
# Apply MinMaxScaler to scale features to a range suitable for VAE (e.g., [0, 1])
# Exclude 'seconds_in_bucket' from scaling                                                                      
features_to_scale = train_data_inliers.columns.difference(['seconds_in_bucket', 'stock_id'])
scaler = MinMaxScaler(feature_range=(0, 1))                                                                    
train_data_scaled = scaler.fit_transform(train_data_inliers[features_to_scale])                                                    
train_data_scaled_df = pd.DataFrame(train_data_scaled, columns=features_to_scale)

train_data_scaled_df['seconds_in_bucket'] = train_data_inliers['seconds_in_bucket'].values
train_data_scaled_df['stock_id'] = train_data_inliers['stock_id'].values

# Drop the corresponding rows from original_data to match train_data_scaled_df                                  
original_data = original_data.loc[train_data_inliers.index] 

# Merge the original dataset with the preprocessed one using 'stock_id' and 'seconds_in_bucket' as keys         
merged_data_cleaned = original_data.merge(train_data_scaled_df, on=['stock_id', 'seconds_in_bucket'])

# Assuming train_data_scaled_df is the preprocessed dataset with 26 features                                    
# Load the original dataset to get the 'stock_id' and 'seconds_in_bucket' columns                               
original_data = pd.read_csv('subset_train.csv')                                      

# Now perform the checks on merged_data                                                                         
# Check for missing time steps within each stock_id                                                             
for stock_id, group in merged_data_cleaned.groupby('stock_id'):                                                         
   # Assuming seconds_in_bucket starts at 0 and increments every 10 seconds                                    
   expected_seconds = list(range(0, group['seconds_in_bucket'].max() + 10, 10))                                
   if not all(second in group['seconds_in_bucket'].values for second in expected_seconds):                     
       print(f"Missing data in stock_id {stock_id}")                                                           
                                                                                                               
# Verify data integrity                                                                                         
if not merged_data_cleaned.sort_values(by=['stock_id', 'seconds_in_bucket']).equals(merged_data_cleaned):                       
   print("Data is not sorted correctly or contains duplicates.")                                               
                                                                                                               
# Determine sequence length based on the frequency of seconds_in_bucket                                         
# For example, if you want sequences representing the last 10 minutes and data is recorded every 10 seconds:    
sequence_length = 10 * 60 // 10  # 60 data points for the last 10 minutes                                       
                                                                                                               
# Check if the sequence length is appropriate given the data frequency                                          
if sequence_length > merged_data_cleaned['seconds_in_bucket'].nunique():                                                
   print("Sequence length is too long for the available time steps.")
