import logging
from agents.shared.data_utils import (
    execute_sql_file,
    upload_parquet_to_bigquery,
)

# Configure logging
logger = logging.getLogger(__name__)

# Import monitoring conditionally
try:
    from utils.monitoring import monitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    monitoring = None


class ModelTrainer:
    def __init__(
        self, project_id, dataset_id, sql_base, table_name=None, data_path=None
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_name = table_name
        self.data_path = data_path
        self.sql_base = sql_base
        logger.info(f"ModelTrainer initialized for project {project_id}, dataset {dataset_id}")

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
        logger.info("Starting model training for all model types")
        for model in ["logistic_reg", "boosted_tree_classifier", "dnn_classifier"]:
            sql_file_name = "train_model.sql"
            if model == "boosted_tree_classifier":
                sql_file_name = "train_model_boosted.sql"

            logger.info(f"Started training model: {model}")
            try:
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
                logger.info(f"Finished training model: {model}")
                if MONITORING_AVAILABLE:
                    monitoring.log_agent_interaction(
                        "model_trainer",
                        "model_trained",
                        {"model": model, "project_id": self.project_id}
                    )
            except Exception as e:
                logger.error(f"Failed to train model {model}: {e}", exc_info=True)
                if MONITORING_AVAILABLE:
                    monitoring.log_error("model_training_error", str(e), {"model": model})
                raise

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
        logger.info(f"Predicting risk profile using model: {model_name}")

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

        logger.debug(f"Prediction parameters: {params}")

        try:
            result = execute_sql_file(
                f"{self.sql_base}/predict_risk_profile.sql",
                params,
                self.project_id,
            )

            if result:
                row = list(result)[0]
                prediction = dict(row)
                logger.info(f"Risk profile prediction successful: {prediction}")
                
                # Only try to log prediction if monitoring is available
                if MONITORING_AVAILABLE and monitoring is not None:
                    try:
                        monitoring.log_prediction(
                            model_name,
                            {
                                "age": age,
                                "education": education,
                                "income": income,
                                "emergency_savings": emergency_savings,
                                "retirement_planning": retirement_planning,
                                "financial_literacy_score": financial_literacy_score
                            },
                            prediction
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log prediction: {e}")
                
                return prediction
            else:
                logger.warning("No prediction result returned from model")
                return None
        except Exception as e:
            logger.error(f"Failed to predict risk profile: {e}", exc_info=True)
            # Only try to log error if monitoring is available
            if MONITORING_AVAILABLE and monitoring is not None:
                try:
                    monitoring.log_error("prediction_error", str(e), {"model": model_name})
                except Exception as log_error:
                    logger.warning(f"Failed to log error: {log_error}")
            raise
