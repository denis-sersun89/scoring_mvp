import json
import numpy as np


class LogisticScoringModel:

    def __init__(self, json_path):
        """
        Загружаем параметры модели из JSON
        """
        with open(json_path, 'r') as f:
            self.config = json.load(f)

        self.model_params = self.config["model"]
        self.variables = self.config["variables"]
        self.master_scale = self.config["master_scale"]

    def get_woe(self, var_name, value):
        """
        Определяет бин и возвращает WoE
        Поддержка numeric и categorical переменных
        """

        bins = self.variables[var_name]["bins"]

        for b in bins:

            # ===============================
            # CATEGORICAL VARIABLE
            # ===============================
            if "values" in b:
                if value in b["values"]:
                    return b["woe"]

            # ===============================
            # NUMERIC VARIABLE
            # ===============================
            else:
                min_val = b.get("min")
                max_val = b.get("max")

                if (min_val is None or value >= min_val) and \
                (max_val is None or value < max_val):
                    return b["woe"]

        raise ValueError(
            f"No bin found for variable {var_name} with value {value}"
        )

    def calculate_logit(self, input_data):
        """
        Рассчитывает логит (это и есть скор)
        """
        intercept = self.model_params["intercept"]
        logit = intercept

        contributions = {}

        for var_name, var_config in self.variables.items():
            value = input_data[var_name]
            woe = self.get_woe(var_name, value)
            coef = var_config["coefficient"]

            contribution = coef * woe
            logit += contribution

            contributions[var_name] = {
                "value": value,
                "woe": woe,
                "coefficient": coef,
                "contribution": contribution
            }

        return logit, contributions

    def calibrate(self, logit):
        """
        Применяем калибровочный intercept и slope
        """
        alpha = self.model_params["calibration_intercept"]
        slope = self.model_params["calibration_slope"]

        calibrated_logit = alpha + slope * logit
        return calibrated_logit

    def calculate_pd(self, calibrated_logit):
        """
        Перевод логита в PD
        """
        return 1 / (1 + np.exp(-calibrated_logit))

    def get_master_rating(self, pd):
        """
        Присваивает рейтинг 1–9
        """
        for r in self.master_scale:
            if r["min_pd"] <= pd < r["max_pd"]:
                return r["rating"]
        return 9

    def score(self, input_data):
        """
        Полный pipeline:
        raw data → logit → calibration → PD → rating
        """
        logit, contributions = self.calculate_logit(input_data)

        calibrated_logit = self.calibrate(logit)
        pd = self.calculate_pd(calibrated_logit)
        rating = self.get_master_rating(pd)

        return {
            "logit_score": logit,
            "calibrated_logit": calibrated_logit,
            "pd": pd,
            "rating": rating,
            "contributions": contributions
        }