from google.cloud import bigquery
from google.api_core.exceptions import NotFound, BadRequest
import traceback


def execute_sql_file(sql_path, params, project_id):
    try:
        with open(sql_path, "r") as file:
            sql = file.read()

        # print("Params:", params)
        # print("SQL:", sql)

        sql = sql.format(**params)
        client = bigquery.Client(project=project_id)
        query_job = client.query(sql)

        return query_job.result()
    except (NotFound, BadRequest) as e:
        print("⚠️ BigQuery execution error (probably missing dataset/table):")
        print(e)
        return None  # Let caller handle the fallback logic

    except Exception as e:
        print("❌ Unexpected error during BigQuery query execution:")
        traceback.print_exc()
        return None


def upload_parquet_to_bigquery(project_id, dataset_id, table_name, parquet_path):
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.{table_name}"
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    with open(parquet_path, "rb") as file:
        load_job = client.load_table_from_file(file, table_id, job_config=job_config)
    load_job.result()


def normalize_value(field: str, value: int) -> int:
    """Normalize special codes 98/99 based on context."""
    if field == "emergency_savings" and value in (98, 99):
        return 2  # Treat as 'Some savings'
    elif field == "retirement_planning" and value in (98, 99):
        return 3  # Treat as 'Unknown'
    elif field in {"age", "education", "income"} and value in (98, 99):
        return 3  # Median
    return value
