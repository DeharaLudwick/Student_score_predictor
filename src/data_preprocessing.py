import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):

    df = pd.read_csv(path)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    df = df.fillna(df.mode().iloc[0])

    # Encode categorical columns
    categorical_columns = df.select_dtypes(include='object').columns

    encoder = LabelEncoder()

    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])

    return df