from google.cloud import bigquery


def execute_sql_file(sql_path, params, project_id):
    with open(sql_path, "r") as file:
        sql = file.read()
    sql = sql.format(**params)
    client = bigquery.Client(project=project_id)
    query_job = client.query(sql)

    result = query_job.result()
    return query_job.result()


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
