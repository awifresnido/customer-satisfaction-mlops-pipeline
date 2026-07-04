"""ZenML data preparation pipeline"""

from zenml import pipeline

from steps.clean_data import clean_data
from steps.ingest_data import ingest_data
from steps.split_data import split_data


@pipeline
def data_pipeline(data_path: str):
    """Execute the complete data preparation pipeline"""

    raw_df = ingest_data(data_path=data_path)
    cleaned_df = clean_data(df=raw_df)
    X_train, X_test, y_train, y_test = split_data(df=cleaned_df)

    return (X_train, X_test, y_train, y_test)
