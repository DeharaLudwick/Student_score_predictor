from src.data_preprocessing import load_and_preprocess
from src.feature_engineering import create_features
from src.train_model import train
from src.evaluate_model import evaluate
from src.visualization import (
    create_heatmap,
    create_feature_importance,
    prediction_plot
)

# Load data
data = load_and_preprocess(
    "data/raw/StudentPerformanceFactors.csv"
)

# Create additional features
data = create_features(data)

# Train model
model, X_test, y_test = train(data)

# Evaluate model
evaluate(
    model,
    X_test,
    y_test
)

# Generate visualizations
create_heatmap(data)

X = data.drop("Exam_Score", axis=1)

create_heatmap(data)

create_feature_importance(
    model,
    X
)

predictions = model.predict(X_test)

prediction_plot(
    y_test,
    predictions
)