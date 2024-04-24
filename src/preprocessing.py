import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Get the directory of the current script
script_directory = os.path.dirname(__file__)

# Define the path to the raw data file
raw_data_path = os.path.join(script_directory, '..', 'artifacts', 'raw_data', 'Churn_Modelling.csv')

# Read the raw data
data = pd.read_csv(raw_data_path)

# Get the unique values in the 'Geography' and 'Gender' columns
unique_geography_values = data['Geography'].unique()
unique_gender_values = data['Gender'].unique()

# Define the preprocessing pipeline
preprocessing_pipeline = ColumnTransformer([
    ('onehot_encoder', OneHotEncoder(drop='first', categories=[unique_geography_values, unique_gender_values]), ['Geography', 'Gender']),
    ('imputer', SimpleImputer(strategy='median'), ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
])

# Define the features and target variable
X = data.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = data['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the full pipeline including preprocessing
full_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline)
])

# Preprocess the training and testing data
X_train_processed = full_pipeline.fit_transform(X_train)
X_test_processed = full_pipeline.transform(X_test)

# Get the column names after preprocessing
preprocessed_column_names = (['Geography_Germany', 'Geography_Spain', 'Gender_Male'] +
                             list(X.drop(columns=['Geography', 'Gender']).columns))

# Ensure that the number of preprocessed columns matches the number of column names
assert X_train_processed.shape[1] == len(preprocessed_column_names), "Number of preprocessed columns does not match the number of column names"

# Create DataFrame objects with the preprocessed data and correct column names
X_train_processed_df = pd.DataFrame(X_train_processed, columns=preprocessed_column_names)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=preprocessed_column_names)

# Define the directory to save the processed data
processed_data_dir = os.path.join(script_directory, '..', 'artifacts', 'processed_data')

# Save the preprocessed training and testing data as CSV files
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=X_train.columns)
X_train_transformed_df.to_csv(os.path.join(processed_data_dir, 'X_train.csv'), index=False)

X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=X_test.columns)
X_test_transformed_df.to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)

# Convert labels to DataFrames
y_train_df = pd.DataFrame(y_train, columns=['is_fraud'])
y_test_df = pd.DataFrame(y_test, columns=['is_fraud'])

# Save the labels as CSV files
y_train_df.to_csv(os.path.join(processed_data_dir, 'y_train.csv'), index=False)
y_test_df.to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False)
