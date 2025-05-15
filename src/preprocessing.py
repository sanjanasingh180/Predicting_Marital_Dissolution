import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path, test_sizes=[0.2, 0.3, 0.4, 0.5]):
    # Load dataset with correct delimiter and skip the first row (question numbers)
    df = pd.read_csv(file_path, sep=";", skiprows=1, header=None)  
    
    print(f"Total rows in dataset after skipping first row: {df.shape[0]}")
    print("Columns in dataset:", df.columns.tolist())
    
    # Assume the last column is the target variable
    target_column = df.columns[-1]  # Dynamically select the last column as target
    print(f"Assumed target column: {target_column}")
    
    # Split features and target variable
    X = df.iloc[:, :-1]  # All columns except last
    y = df.iloc[:, -1]   # Last column as target
    
    results = {}
    
    for test_size in test_sizes:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y  # Stratify to maintain class ratio
        )

        print(f"Split {int((1-test_size)*100)}-{int(test_size*100)}: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
        
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        results[f"train_{int((1-test_size)*100)}_test_{int(test_size*100)}"] = {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train_resampled,
            "y_test": y_test
        }
    
    return results

if __name__ == "__main__":
    file_path = "data/divorce_data.csv"  
    data_splits = load_and_preprocess_data(file_path)
    
    for key, data in data_splits.items():
        print(f"✅ Data preprocessing complete for {key}. Shapes: {data['X_train'].shape}, {data['X_test'].shape}")
