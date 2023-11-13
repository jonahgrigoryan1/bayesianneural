from sklearn.impute import KNNImputer                                                                           
                                                                                                                   
# Create an instance of KNNImputer                                                                              
imputer = KNNImputer(n_neighbors=5)                                                                             
                                                                                                                   
# Fit the imputer to the dataset and transform it                                                               
train_data_imputed = imputer.fit_transform(train_data)                                                          
                                                                                                                   
# Convert the array back to a pandas DataFrame                                                                  
train_data_imputed = pd.DataFrame(train_data_imputed, columns=train_data.columns)                               

# Check for missing values again                                                                                
missing_values_per_column_after_imputation = train_data_imputed.isnull().sum()                                  
total_missing_values_after_imputation = train_data_imputed.isnull().sum().sum()                                 
                                                                                                                   
missing_values_per_column_after_imputation, total_missing_values_after_imputation  
