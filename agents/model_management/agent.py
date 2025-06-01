from agents.model_management.utils import execute_sql_file, upload_parquet_to_bigquery
from dotenv import load_dotenv
import os


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


class ModelManagementAgent:
    """An agent responsible for:

    1. Creating the BigQuery dataset if it doesn’t exist.
    2. Uploading cleaned training data (from Parquet).
    3. Evaluating multiple model types (`LOGISTIC_REG`, `BOOSTED_TREE_CLASSIFIER`, `DNN_CLASSIFIER`, etc.).
    4. Selecting the best model based on evaluation metrics.
    5. Training and storing the final model for use by other agents."""

    def __init__(self, project_id, dataset_id, data_path, table_name):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.data_path = data_path
        self.table_name = table_name
        self.sql_base = "agents/model_management/sql/"

    def setup(self):
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

    def evaluate_models(self):
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
            print(f"Evaluation for {model}:", list(result))

    def run(self):
        self.setup()
        self.upload_data()
        self.train_models()
        self.evaluate_models()


if __name__ == "__main__":
    # print(os.getenv("GCP_PROJECT_ID"))
    agent = ModelManagementAgent(
        project_id=os.getenv("GCP_PROJECT_ID"),
        dataset_id=os.getenv("BQ_DATASET"),
        data_path=os.getenv("DATA_PATH"),
        table_name=os.getenv("BQ_TABLE"),
    )
    agent.run()
