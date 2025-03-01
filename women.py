import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
import joblib

# Load datasets
def load_data():
    """Load the main health dataset and women-specific dataset."""
    try:
        df_main = pd.read_csv("Dataset - Updated.csv")
        df_women = pd.read_csv("women-spec-health.csv")
        return df_main, df_women
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None, None

# Standardize column names
def standardize_column_names(df):
    """Strip any extra spaces in column names."""
    df.columns = df.columns.str.strip()

def rename_columns(df_women):
    """Rename columns for better clarity and consistency."""
    df_women.rename(columns={
        'Pcosn': 'PCOS',
        'A1': 'Menstrual Irregularities',
        'A2': 'Menopause',
        'A3': 'Pregnancy Status'
    }, inplace=True)

def preprocess_women_data(df_women):
    """Preprocess women-specific health data, converting categorical to numerical."""
    if 'Age' not in df_women.columns:
        raise KeyError("❌ 'Age' column not found in women-specific dataset.")

    df_women = df_women[['Age', 'PCOS', 'Menstrual Irregularities', 'Menopause', 'Pregnancy Status']]

    # Convert categorical values to numerical
    for col in ['PCOS', 'Menstrual Irregularities', 'Menopause', 'Pregnancy Status']:
        df_women[col] = df_women[col].map({'Yes': 1, 'No': 0, 'Pregnant': 1, 'Not Pregnant': 0}).fillna(0)
    
    return df_women

# Merge main dataset with women-specific data
def merge_datasets(df_main, df_women):
    """Merge the main dataset with the women-specific dataset on 'Age'."""
    return pd.merge(df_main, df_women, on='Age', how='left')

# Prepare features and target
def prepare_data(df):
    """Prepare features (X) and target (y) for machine learning."""
    FEATURES = [
        'Age', 'Systolic BP', 'Diastolic', 'BS (Blood Sugar)', 'Body Temp', 'BMI', 
        'Previous Complications', 'Preexisting Diabetes', 'Gestational Diabetes', 
        'Mental Health', 'Heart Rate', 'PCOS', 'Menstrual Irregularities', 'Menopause', 'Pregnancy Status'
    ]
    TARGET = 'Risk Level'

    df.rename(columns={"BS": "BS (Blood Sugar)"}, inplace=True)

    df['Risk Level'] = df['Risk Level'].map({'High': 1, 'Low': 0}).fillna(0).astype(int)

    # Handle missing values by filling with median values for numeric columns
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df[FEATURES].copy()
    y = df[TARGET]

    return X, y

def standardize_features(X):
    """Standardize the features using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for future predictions
    joblib.dump(scaler, "scaler.pkl")
    
    return X_scaled

def train_model(X_train, y_train):
    """Train a Random Forest Classifier model."""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Save the trained model for future predictions
    joblib.dump(rf_model, "women_risk_model.pkl")
    
    return rf_model

# Perform Hierarchical Clustering
def apply_hierarchical_clustering(X_scaled):
    """Apply Hierarchical Clustering to the dataset."""
    try:
        hierarchy = linkage(X_scaled, method='ward')
        return fcluster(hierarchy, t=3, criterion='maxclust')
    except Exception as e:
        print(f"⚠️ Clustering failed: {e}")
        return np.full(X_scaled.shape[0], -1)  # Default value if clustering fails

# Perform Gaussian Mixture Model (GMM) clustering
def apply_gmm_clustering(X_scaled):
    """Apply Gaussian Mixture Model (GMM) clustering to the dataset."""
    try:
        gmm = GaussianMixture(n_components=3, random_state=42)
        return gmm.fit_predict(X_scaled)
    except Exception as e:
        print(f"⚠️ GMM Clustering failed: {e}")
        return np.full(X_scaled.shape[0], -1) 

def save_clustered_data(df):
    """Save the dataset with cluster labels to CSV."""
    df.to_csv("women_clustered_data.csv", index=False)

def main():
    """Main function to execute the data processing and model training pipeline."""
    df_main, df_women = load_data()
    if df_main is None or df_women is None:
        return

    standardize_column_names(df_main)
    standardize_column_names(df_women)

    rename_columns(df_women)
    df_women = preprocess_women_data(df_women)

    df = merge_datasets(df_main, df_women)

    X, y = prepare_data(df)

    X_scaled = standardize_features(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the Random Forest Classifier model
    rf_model = train_model(X_train, y_train)

    # Apply Hierarchical Clustering
    df['Hierarchical Cluster'] = apply_hierarchical_clustering(X_scaled)

    # Apply Gaussian Mixture Model (GMM) Clustering
    df['GMM Cluster'] = apply_gmm_clustering(X_scaled)

    save_clustered_data(df)

    print("✅ Model training and clustering complete. Files saved:")
    print(" - women_risk_model.pkl (Risk Prediction Model)")
    print(" - women_clustered_data.csv (Clustered Data for Personalized Recommendations)")
    print(" - scaler.pkl (StandardScaler for prediction)")

if __name__ == "__main__":
    main()
