import matplotlib.pyplot as plt
import seaborn as sns


def create_heatmap(data):

    plt.figure(figsize=(12,8))

    sns.heatmap(
        data.corr(),
        annot=True,
        cmap='coolwarm'
    )

    plt.title("Correlation Heatmap")

    plt.savefig(
        "images/heatmap.png"
    )

    plt.show()


def create_feature_importance(model, X):

    importance = model.feature_importances_

    plt.figure(figsize=(10,6))

    plt.bar(
        X.columns,
        importance
    )

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.savefig(
        "images/feature_importance.png"
    )

    plt.show()



def prediction_plot(y_test, predictions):

    plt.figure(figsize=(8,6))

    plt.scatter(
        y_test,
        predictions
    )

    plt.xlabel("Actual Scores")
    plt.ylabel("Predicted Scores")

    plt.savefig(
        "images/prediction_plot.png"
    )

    plt.show()