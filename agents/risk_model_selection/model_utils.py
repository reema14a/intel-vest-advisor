from agents.risk_model_selection.utils import (
    execute_sql_file,
    upload_parquet_to_bigquery,
)


class ModelTrainer:
    def __init__(self, project_id, dataset_id, table_name, data_path, sql_base):
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

    def evaluate_all_models(self):
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

    def train_all_models(self):
        self.setup()
        self.train_models()
