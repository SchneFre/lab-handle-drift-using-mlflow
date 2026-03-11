import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Start MLflow run
with mlflow.start_run():

    # Load datasets
    reference_data = pd.read_csv("reference_data.csv")
    current_data = pd.read_csv("current_data.csv")

    # Create Evidently drift report
    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    # Save HTML report
    report_path = "drift_report.html"
    report.save_html(report_path)

    # Log report to MLflow
    mlflow.log_artifact(report_path)

    print("Drift report generated and logged to MLflow.")