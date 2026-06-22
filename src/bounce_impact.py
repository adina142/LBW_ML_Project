import numpy as np

def detect_bounce_point(tracking_df):

    idx = np.argmax(
        tracking_df["Kalman_Y"]
    )

    return {
        "Pitch_X":
        tracking_df.iloc[idx]["Kalman_X"],

        "Pitch_Y":
        tracking_df.iloc[idx]["Kalman_Y"]
    }

def detect_impact_point(tracking_df):

    row = tracking_df.iloc[-1]

    return {
        "Impact_X": row["Kalman_X"],
        "Impact_Y": row["Kalman_Y"]
    }