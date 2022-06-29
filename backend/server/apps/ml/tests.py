from django.test import TestCase

from apps.ml.matchwinner.random_forest import RandomForestClassifier



import inspect
from apps.ml.registry import MLRegistry

class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = {
            "Map":3,
            "Team":0,
            "InternalTeamId":1,
            "MatchId":4,
            "RoundId":1,
            "SteamId":76561197971812216,
            "RoundWinner":0,
            "Survived":0,
            "TimeAlive":43.486626,
            "ScaledTimeAlive":0.81861,
            "AvgCentroidDistance":547137.458474,
            "TravelledDistance":60960.883185,
            "AvgRoundVelocity":87.839889,
            "AvgKillDistance":0.0,
            "AvgSiteDistance":5.415945e+06,
            "RLethalGrenadesThrown":0,
            "RNonLethalGrenadesThrown":0,
            "PrimaryAssaultRifle":0.0,
            "PrimarySniperRifle":0.0,
            "PrimaryHeavy":0.0,
            "PrimarySMG":0.0,
            "PrimaryPistol":1,
            "FirstKillTime":0.0,
            "RoundKills":0,
            "RoundAssists":0,
            "RoundHeadshots":0,
            "RoundFlankKills":0,
            "RoundStartingEquipmentValue":800,
            "TeamStartingEquipmentValue":4400
            }
        my_alg = RandomForestClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual(0, response['label'])

# add below method to MLTests class:
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "income_classifier"
        algorithm_object = RandomForestClassifier()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Piotr"
        algorithm_description = "Random Forest with simple pre- and post-processing"
        algorithm_code = inspect.getsource(RandomForestClassifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
