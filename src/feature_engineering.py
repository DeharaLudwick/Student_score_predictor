def create_features(df):

    if 'Hours_Studied' in df.columns and 'Attendance' in df.columns:

        df["Study_Efficiency"] = (
            df["Hours_Studied"] *
            df["Attendance"]
        )

    return df