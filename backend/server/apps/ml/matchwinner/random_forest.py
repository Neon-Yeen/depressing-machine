# file backend/server/apps/ml/income_classifier/random_forest.py
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

class RandomForestClassifier:
    def __init__(self):
        path_to_artifacts = "../../research/"
        #self.values_fill_missing =  joblib.load(path_to_artifacts + "train_mode.joblib")
        #self.encoders = joblib.load(path_to_artifacts + "encoders.joblib")
        self.model = joblib.load(path_to_artifacts + "random_forest.joblib")

    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        # fill missing values
        #input_data.fillna(self.values_fill_missing)
        # convert categoricals
        for column in [
            "Map",
            "Team",
            "InternalTeamId",
            "MatchId",
            "RoundId",
            "SteamId",
            "RoundWinner",
            "Survived",
            "TimeAlive",
            "ScaledTimeAlive",
            "AvgCentroidDistance",
            "TravelledDistance",
            "AvgRoundVelocity",
            "AvgKillDistance",
            "AvgSiteDistance",
            "RLethalGrenadesThrown",
            "RNonLethalGrenadesThrown",
            "PrimaryAssaultRifle",
            "PrimarySniperRifle",
            "PrimaryHeavy",
            "PrimarySMG",
            "PrimaryPistol",
            "FirstKillTime",
            "RoundKills",
            "RoundAssists",
            "RoundHeadshots",
            "RoundFlankKills",
            "RoundStartingEquipmentValue",
            "TeamStartingEquipmentValue"

        ]:
            #categorical_convert = self.encoders[column]
            input_data[column] = input_data[column]#StandardScaler().fit_transform(input_data[column])

        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    def postprocessing(self, input_data):
        label = 0
        if input_data[1] == 1:
            label = 1
        return {"probability": input_data[1], "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
