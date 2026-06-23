# core/ml_model.py
import joblib
import numpy as np

class PostureModel:
    def __init__(self, path="model.pkl"):
        try:
            self.model = joblib.load(path)
        except:
            self.model = None

    def predict(self, features):
        if self.model is None:
            return None, 0.0

        x = [
            features.get("left_knee", 0),
            features.get("right_knee", 0),
            features.get("hip_angle", 0),
            features.get("sway_velocity", 0),
            features.get("jump_feet", 0),
        ]

        x = np.array(x).reshape(1, -1)

        pred = self.model.predict(x)[0]

        if hasattr(self.model, "predict_proba"):
            conf = max(self.model.predict_proba(x)[0])
        else:
            conf = 0.5

        return pred, conf