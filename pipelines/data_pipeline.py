from zenml import pipeline

from steps.ingest_data import ingest_data


@pipeline
def data_pipeline(data_path: str):

    ingest_data(data_path=data_path)
