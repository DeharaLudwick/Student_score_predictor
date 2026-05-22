from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def evaluate(model,X_test,y_test):

    predictions=model.predict(X_test)

    print(
        "MAE:",
        mean_absolute_error(
            y_test,
            predictions
        )
    )

    print(
        "RMSE:",
        mean_squared_error(
            y_test,
            predictions
        )**0.5
    )

    print(
        "R²:",
        r2_score(
            y_test,
            predictions
        )
    )