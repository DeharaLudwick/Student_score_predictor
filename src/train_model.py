from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train(df):

    X = df.drop("Exam_Score", axis=1)

    y = df["Exam_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train,y_train)

    joblib.dump(
        model,
        "models/student_score_model.pkl"
    )

    return model,X_test,y_test