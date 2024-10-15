import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your DataFrame (assuming it's saved as a CSV for the app)
@st.cache_data
def load_data():
    return pd.read_csv('dataset\\ConsumerElectronics.csv')

ce_df = load_data()

# Data Conversion Function
def data_conversion(df):
    df = df.replace('\\N', pd.NA)
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]

    df['deliverybdays'] = pd.to_numeric(df['deliverybdays'], errors='coerce')
    df['deliverycdays'] = pd.to_numeric(df['deliverycdays'], errors='coerce')
    df = df[(df['deliverybdays'] >= 1) & (df['deliverycdays'] >= 1)]
    columns_to_keep = df.nunique()[df.nunique() > 1].index
    df = df[columns_to_keep]
    df['order_date'] = pd.to_datetime(df['order_date'])
    df["gmv"] = pd.to_numeric(df["gmv"], errors='coerce')
    df["gmv"].fillna(df["gmv"].median(), inplace=True)
    df = df.drop_duplicates()

    return df

# Apply data conversion
ce_df = data_conversion(ce_df)

# Check for duplicates
duplicates = ce_df.duplicated()
num_duplicates = duplicates.sum()

# Frequency Encoding for Categorical Columns
categorical_df = ce_df.select_dtypes(include=['object'])
for col in categorical_df.columns:
    freq_encoding = categorical_df[col].value_counts(normalize=True)
    categorical_df[col] = categorical_df[col].map(freq_encoding)

# Combine with Numeric Data
numeric_df = ce_df.select_dtypes(include=['int64', 'float64'])
combined_df = pd.concat([numeric_df, categorical_df], axis=1)

# Correlation Matrix
correlation_matrix = combined_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
st.pyplot(plt)

# Add a sidebar option for the type of analysis (univariate or bivariate)
analysis_type = st.sidebar.selectbox('Select the type of analysis', ['Univariate Analysis', 'Bivariate Analysis'])

if analysis_type == 'Univariate Analysis':
    st.subheader('Univariate Analysis')
    
    # Add a column selector for numeric columns in univariate analysis
    numeric_columns = ce_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_numeric_columns = st.multiselect('Select numeric columns for univariate analysis', numeric_columns, default=['gmv', 'deliverybdays', 'deliverycdays', 'product_mrp', 'sla'])
    
    # Plot histograms for selected numeric columns
    ce_df[selected_numeric_columns].hist(bins=30, figsize=(15, 10))
    st.pyplot(plt)

    # Add a column selector for categorical columns
    categorical_columns = ce_df.select_dtypes(include=['object']).columns.tolist()
    selected_categorical_columns = st.multiselect('Select categorical columns for count plots', categorical_columns, default=['s1_fact.order_payment_type', 'product_analytic_category', 'product_analytic_sub_category'])
    
    # Plot count plots for selected categorical columns
    for col in selected_categorical_columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=ce_df, x=col)
        plt.title(f'Count Plot of {col}')
        plt.xticks(rotation=90)
        st.pyplot(plt)

elif analysis_type == 'Bivariate Analysis':
    st.subheader('Bivariate Analysis')

    # Add a target column selector (default to 'gmv')
    target_column = st.selectbox('Select target column for bivariate analysis', ce_df.columns, index=ce_df.columns.get_loc('gmv'))

    # Add a column selector for other columns, excluding the target column
    other_columns = ce_df.columns.tolist()
    other_columns.remove(target_column)
    selected_bivariate_columns = st.multiselect('Select columns for bivariate analysis', other_columns, default=['deliverybdays', 'product_mrp', 'sla', 'product_analytic_category'])

    # Bivariate Analysis Function
    def comprehensive_bivariate_analysis(df, target_col, selected_cols):
        sns.set(style="whitegrid")
        base_palette = sns.color_palette("Set2")

        for col in selected_cols:
            plt.figure(figsize=(15, 5))
            if df[col].dtype in ['int64', 'float64'] and df[target_col].dtype in ['int64', 'float64']:
                plt.subplot(1, 2, 1)
                sns.scatterplot(data=df, x=col, y=target_col, marker='o', s=100, edgecolor='k')
                plt.title(f'Scatter Plot: {target_col} vs {col}')
                plt.xlabel(col)
                plt.ylabel(target_col)
                plt.grid(True, linestyle='--')

                plt.subplot(1, 2, 2)
                sns.heatmap(df[[col, target_col]].corr(), annot=True, cmap='coolwarm', center=0, linewidths=1, linecolor='k')
                plt.title('Correlation Heatmap')
                st.pyplot(plt)

            elif df[col].dtype == 'object' and df[target_col].dtype in ['int64', 'float64']:
                unique_vals = df[col].nunique()
                custom_palette = sns.color_palette(base_palette[:unique_vals]) if unique_vals <= len(base_palette) else sns.color_palette("husl", unique_vals)

                plt.subplot(1, 2, 1)
                sns.boxplot(data=df, x=col, y=target_col, palette=custom_palette, hue=col, dodge=False)
                plt.title(f'Box Plot: {target_col} by {col}')
                plt.xlabel(col)
                plt.ylabel(target_col)
                plt.xticks(rotation=90)
                plt.grid(True, linestyle='--')
                plt.legend([],[], frameon=False)

                plt.subplot(1, 2, 2)
                sns.countplot(data=df, x=col, palette=custom_palette, hue=col, dodge=False)
                plt.title(f'Count Plot: {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.grid(True, linestyle='--')
                plt.xticks(rotation=90)
                plt.legend([],[], frameon=False)
                st.pyplot(plt)

            else:
                st.warning(f"Unsupported data types for the specified columns: {col} and {target_col}.")

    # Call the bivariate analysis function with selected columns
    comprehensive_bivariate_analysis(ce_df, target_column, selected_bivariate_columns)

# Split the data into training and testing sets
X = ce_df.drop(columns='gmv')  # Features
y = ce_df['gmv']                # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create copies for encoding
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

# Log transform the target variable in the training set
y_train_log = np.log1p(y_train)  # Apply log transformation to the training target
# You can log-transform the test set if needed for analysis, but typically you would keep the original values for prediction
y_test_log = np.log1p(y_test)  # Optional: apply log transform for analysis

# Identify categorical columns except for 'product_analytic_vertical'
categorical_cols = X_train_encoded.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('product_analytic_vertical')

# Apply Label Encoding to all categorical columns except for 'product_analytic_vertical'
# Apply Label Encoding to all categorical columns except for 'product_analytic_vertical'
label_encoder = LabelEncoder()

for col in categorical_cols:
    # Fit the label encoder on the training data
    X_train_encoded[col] = label_encoder.fit_transform(X_train_encoded[col])
    
    # For the test data, check for unseen labels and handle them
    X_test_encoded[col] = X_test[col].apply(
        lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1
    )


# Apply Target Encoding to 'product_analytic_vertical'
target_encoder = TargetEncoder()
X_train_encoded['product_analytic_vertical'] = target_encoder.fit_transform(
    X_train_encoded['product_analytic_vertical'], 
    y_train_log  # Use the log-transformed training target variable for encoding
)

X_test_encoded['product_analytic_vertical'] = target_encoder.transform(X_test_encoded['product_analytic_vertical'])

# Convert any datetime columns to numeric (Unix timestamps)
for col in X_train_encoded.select_dtypes(include=['datetime64[ns]']).columns:
    X_train_encoded[col] = X_train_encoded[col].astype(np.int64) // 10**9

for col in X_test_encoded.select_dtypes(include=['datetime64[ns]']).columns:
    X_test_encoded[col] = X_test_encoded[col].astype(np.int64) // 10**9

# Initialize the MinMaxScaler
minmax_scaler = MinMaxScaler()

# Select numeric columns for normalization
numeric_cols = X_train_encoded.select_dtypes(include=[np.number]).columns.tolist()

# Normalize the training and testing data
X_train_normalized = X_train_encoded.copy()
X_train_normalized[numeric_cols] = minmax_scaler.fit_transform(X_train_encoded[numeric_cols])

X_test_normalized = X_test_encoded.copy()
X_test_normalized[numeric_cols] = minmax_scaler.transform(X_test_encoded[numeric_cols])

# Train the Random Forest Regressor model
if st.button('Train Random Forest Model'):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_normalized, y_train_log)

    # Make predictions
    y_pred_log = model.predict(X_test_normalized)

    # Inverse log transformation
    y_pred = np.expm1(y_pred_log)
    y_test = np.expm1(y_test_log)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate adjusted R²
    n = len(y_test)
    p = X_test_normalized.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# --- Prediction Section ---
st.subheader('Prediction on New Data')
uploaded_file = st.file_uploader("Upload new data for predictions (CSV file)", type="csv")

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    
    # Apply the same preprocessing steps
    new_data_encoded = new_data.copy()
    for col in categorical_cols:
        new_data_encoded[col] = label_encoder.transform(new_data[col])

    new_data_encoded['product_analytic_vertical'] = target_encoder.transform(new_data_encoded['product_analytic_vertical'])

    for col in new_data_encoded.select_dtypes(include=['datetime64[ns]']).columns:
        new_data_encoded[col] = new_data_encoded[col].astype(np.int64) // 10**9

    new_data_normalized = new_data_encoded.copy()
    new_data_normalized[numeric_cols] = minmax_scaler.transform(new_data_encoded[numeric_cols])

    # Make predictions
    new_predictions_log = model.predict(new_data_normalized)
    new_predictions = np.expm1(new_predictions_log)

    # Display predictions
    st.write("Predicted GMV for uploaded data:")
    st.write(new_predictions)