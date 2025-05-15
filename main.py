from src.preprocessing import load_and_preprocess_data
from src.train import train_models
from src.evaluate import evaluate_all_models

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/divorce_data.csv")

# Train models
trained_models = train_models(X_train, y_train)

# Evaluate models
results = evaluate_all_models(X_test, y_test)

# Print evaluation results
for model, metrics in results.items():
    print(f"\n{model} Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

