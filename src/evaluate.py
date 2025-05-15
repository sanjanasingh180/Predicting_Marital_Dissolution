import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

# Load preprocessed data
from src.preprocessing import load_and_preprocess_data

def train_and_evaluate_models(file_path, test_sizes=[0.2, 0.3, 0.4, 0.5]):
    # Load the preprocessed data
    data_splits = load_and_preprocess_data(file_path, test_sizes=test_sizes)
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }
    
    results = []  # Store evaluation results
    
    for test_size in test_sizes:
        key = f"train_{int((1-test_size)*100)}_test_{int(test_size*100)}"
        X_train, X_test = data_splits[key]['X_train'], data_splits[key]['X_test']
        y_train, y_test = data_splits[key]['y_train'], data_splits[key]['y_test']
        
        print(f"\n📊 Evaluating models for {key} split...")
        
        best_model = None
        best_accuracy = 0
        
        for name, model in models.items():
            print(f"\nTraining {name} model...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            report = classification_report(y_test, y_pred)
            
            print(f"{name} Model Accuracy: {accuracy:.4f}")
            print(f"Classification Report for {name}:\n", report)
            
            # Save the trained model
            model_filename = f"trained_model_{name}_{key}.pkl"
            with open(model_filename, "wb") as model_file:
                pickle.dump(model, model_file)
            
            print(f"✅ {name} model training complete and saved as '{model_filename}'")
            
            # Store results
            results.append([key, name, accuracy, precision, recall, f1])
            
            # Track the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = name
        
        print(f"\n🏆 Best Model for {key}: {best_model} with Accuracy: {best_accuracy:.4f}")
    
    # Save results as a DataFrame and CSV
    results_df = pd.DataFrame(results, columns=["Split", "Model", "Accuracy", "Precision", "Recall", "F1-score"])
    results_df.to_csv("model_evaluation_results.csv", index=False)
    print("📄 Model evaluation results saved to 'model_evaluation_results.csv'")

if __name__ == "__main__":
    file_path = "data/divorce_data.csv"  # Update the correct path if needed
    train_and_evaluate_models(file_path)
