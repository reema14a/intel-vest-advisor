from agents.shared.data_utils import (
    execute_sql_file,
    upload_parquet_to_bigquery,
)


class ModelTrainer:
    def __init__(
        self, project_id, dataset_id, sql_base, table_name=None, data_path=None
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_name = table_name
        self.data_path = data_path
        self.sql_base = sql_base

    def create_dataset(self):
        execute_sql_file(
            f"{self.sql_base}/create_dataset.sql",
            {"project_id": self.project_id, "dataset_id": self.dataset_id},
            self.project_id,
        )

    def upload_data(self):
        upload_parquet_to_bigquery(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            table_name=self.table_name,
            parquet_path=self.data_path,
        )

    def setup(self):
        self.create_dataset()
        self.upload_data()

    def train_models(self):
        for model in ["logistic_reg", "boosted_tree_classifier", "dnn_classifier"]:
            sql_file_name = "train_model.sql"
            if model == "boosted_tree_classifier":
                sql_file_name = "train_model_boosted.sql"

            print(f"Started training model: {model}")
            execute_sql_file(
                f"{self.sql_base}/{sql_file_name}",
                {
                    "project_id": self.project_id,
                    "dataset_id": self.dataset_id,
                    "table_name": self.table_name,
                    "model_type": model,
                },
                self.project_id,
            )
            print(f"Finished training model: {model}")

    def train_all_models(self):
        self.setup()
        self.train_models()

    def evaluate_all_models(self):
        print("Evaluating all models...")
        # print(f"Project ID: {self.project_id}")
        # print(f"Dataset ID: {self.dataset_id}")
        # print(f"SQL Base: {self.sql_base}")

        evaluations = []

        for model in ["logistic_reg", "boosted_tree_classifier", "dnn_classifier"]:
            result = execute_sql_file(
                f"{self.sql_base}/evaluate_model.sql",
                {
                    "project_id": self.project_id,
                    "dataset_id": self.dataset_id,
                    "model_name": f"model_{model}",
                },
                self.project_id,
            )

            result_list = list(result)
            print(f"Evaluation for {model}:", result_list)

            evaluations.extend(result_list)

        if result is None:
            return None

        print("All evaluations:", evaluations)
        return evaluations

    def predict_risk_profile(
        self,
        model_name,
        age,
        education,
        income,
        emergency_savings,
        retirement_planning,
        financial_literacy_score,
    ):
        print("Predicting risk profile...")
        print(f"Model: {model_name}")

        params = {
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "model_name": f"model_{model_name}",  # model name
            "age_group": age,
            "education_level": education,
            "income_group": income,
            "emergency_savings": emergency_savings,
            "retirement_planning": retirement_planning,
            "financial_literacy_score": financial_literacy_score,
        }

        result = execute_sql_file(
            f"{self.sql_base}/predict_risk_profile.sql",
            params,
            self.project_id,
        )

        if result:
            row = list(result)[0]
            return dict(row)
        return None
