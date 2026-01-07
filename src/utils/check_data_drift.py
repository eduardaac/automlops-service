import pandas as pd
import json

from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.metrics import DatasetDriftMetric
from src.utils.data_utils import remove_target_columns

def check_data_drift(reference, current):
    """
    Check data drift between two datasets: reference and current.

    Uses Evidently library to generate a report on data drift
    between reference and current datasets. Comparison is made at both
    global dataset level and per column.

    Args:
        reference (pd.DataFrame): Reference dataset.
        current (pd.DataFrame): Current dataset for comparison.

    Returns:
        dict: Dictionary containing data drift information between datasets,
              including number of drifted columns, mean drift score and details
              of drift per column.
    """
    
    # Remover colunas de target antes de checar drift (usando utilitário centralizado)
    ref = remove_target_columns(reference.copy())
    cur = remove_target_columns(current.copy())

    data_drift_dataset_report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])

    data_drift_dataset_report.run(reference_data=ref, current_data=cur)

    report_json = json.loads(data_drift_dataset_report.json())['metrics'][1]['result']

    print(data_drift_dataset_report.json())

    res = {}

    res['number_of_columns'] = report_json['number_of_columns']
    res['number_of_drifted_columns'] = report_json['number_of_drifted_columns']
    res['mean_drifted_score'] = report_json['share_of_drifted_columns']
    res['dataset_drift'] = report_json['dataset_drift']

    cols = {}
    drift_by_columns = report_json['drift_by_columns']
    
    for column_name, column_info in drift_by_columns.items():
        drift_detected = column_info['drift_detected']
        drift_score = column_info['drift_score']
        cols[column_name] = {'drift_detected': drift_detected, 'drift_score': drift_score}
        
    res['columns'] = cols

    return res
