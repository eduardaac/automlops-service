import os
import logging
from os.path import exists
from typing import List
import pandas as pd
from pycaret.classification import ClassificationExperiment
from src.classes.Model import Model

class Automl:
    """
    Automated Machine Learning (AutoML) class for model selection and tuning.

    This class uses the PyCaret library for automated machine learning experimentation,
    including model selection, hyperparameter tuning, and API generation.

    Args:
        dataset: Input dataset for machine learning.
        filename (str): Dataset file name.
        metric (str, optional): Evaluation metric used for model selection and tuning. Default is "Accuracy".
        target_position (int, optional): Position of the target variable in the dataset. Default is -1.

    Attributes:
        filename (str): Dataset file name.
        metric (str): Evaluation metric used for model selection and tuning.
        _target_position (int): Position of the target variable in the dataset.
        _dataset: Input dataset for machine learning.
        _cs: PyCaret ClassificationExperiment instance for AutoML configuration.
        _setup: PyCaret configuration for the AutoML experiment.

    Methods:
        find_best_models(n: int) -> List[Model]: Select the 'n' best machine learning models based on specified metric.
        tuning_model(model: Model) -> Model: Tune hyperparameters of a given machine learning model.
        save_model(model: Model, model_name: str = "") -> str: Save a trained machine learning model to a file.
        eval_model(model: Model) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
            Evaluate a machine learning model on training and test sets.
        generate_api(model: Model, api_name: str = "", host: str = "127.0.0.1", port: int = 5000) -> str:
            Generate a FastAPI script to serve the machine learning model as an API.

    Note:
        This class assumes the use of the PyCaret library for AutoML functionalities.
    """

    def __init__(
        self,
        dataset,
        filename: str,
        metric: str = "Accuracy",
        target_position: int = -1,
    ) -> None:
        """
        Initialize Automl instance for automated machine learning.

        Args:
            dataset: Input dataset for machine learning.
            filename (str): Dataset file name.
            metric (str, optional): Evaluation metric used for model selection and tuning. Default is "Accuracy".
            target_position (int, optional): Position of the target variable in the dataset. Default is -1.
        """
        self.filename = filename
        self.metric = metric
        self._target_position = target_position
        self._dataset = dataset
        self._cs = ClassificationExperiment()
        self._setup = self._cs.setup(
            data=self._dataset,
            target=self._target_position,
            train_size=0.7,
            numeric_imputation="knn",
            categorical_imputation="mode",
            html=False,
            verbose=False,
            fold_strategy="kfold",
            fold=3,
            fix_imbalance=True,
            remove_outliers=True,
            outliers_method="iforest",
            session_id=None,
            n_jobs=2,
        )
        print(self._setup.get_config())

    def find_best_models(self, n: int) -> List[Model]:
        """
        Select the 'n' best machine learning models based on the specified metric.

        Args:
            n (int): Number of top models to select.

        Returns:
            List[Model]: List of 'n' selected machine learning models.
        """
        top_models = self._setup.compare_models(
            sort=self.metric, n_select=n, verbose=False, exclude=["dummy"]
        )
        
        if n == 1:
            if not isinstance(top_models, list):
                top_models = [top_models]
        
        return [Model(estimator) for estimator in top_models]

    def tuning_model(self, model: Model) -> Model:
        """
        Tune hyperparameters of a given machine learning model.

        Args:
            model (Model): Machine learning model to tune.

        Returns:
            Model: Tuned machine learning model.
        """
        tuned_model = self._setup.tune_model(
            model.get_estimator(),
            n_iter=3,
            search_library="scikit-learn",
            search_algorithm="random",
            optimize=self.metric,
            verbose=False,
        )
        return Model(tuned_model)

    def save_model(self, model: Model, model_name: str = ""):
        """
        Save a trained machine learning model to a file.

        Args:
            model (Model): Trained machine learning model to save.
            model_name (str, optional): Name to use for the saved model file.
                If not provided, the model's readable name will be used.

        Returns:
            str: Path to the saved model file.
        """
        logger = logging.getLogger(__name__)
        
        if model_name == "":
            model_name = model.get_model_name()
        pickle_file = f"./tmp/files/{self.filename.split('.')[0]}_{model_name}"
        logger.info(f"Pickle file path: {pickle_file}")
        print(pickle_file)

        if exists(pickle_file):
            pickle_file = f"{pickle_file}_new"
            logger.info(f"File already exists, using new name: {pickle_file}")

        try:
            saved_model = self._setup.save_model(
                model.get_estimator(), model_name=pickle_file, model_only=True
            )
            logger.info(f"PyCaret save_model executed successfully. Return: {saved_model}")
        except Exception as e:
            logger.error(f"Error in PyCaret save_model: {str(e)}")
            raise e
            
        return pickle_file

    def eval_model(self, model: Model):
        """
        Evaluate a machine learning model on training and test sets.

        Args:
            model (Model): Machine learning model to evaluate.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
                Tuple containing X_train, y_train, X_test, y_test, and predictions DataFrame.
        """
        x_train = self._setup.get_config("X_train")
        y_train = self._setup.get_config("y_train")
        x_test = self._setup.get_config("X_test")
        y_test = self._setup.get_config("y_test")
        return (
            x_train,
            y_train,
            x_test,
            y_test,
            self._setup.predict_model(model.get_estimator()),
        )

    def generate_api(
        self,
        model: Model,
        api_name: str = "",
        host: str = "127.0.0.1",
        port: int = 5000,
    ) -> str:
        """
        Generate a FastAPI script to serve the machine learning model as an API.

        Args:
            model (Model): Machine learning model to serve via API.
            api_name (str, optional): Name to use for the generated FastAPI script.
                If not provided, a name based on the dataset and model will be used.
            host (str, optional): Host address for the FastAPI server. Default is "127.0.0.1".
            port (int, optional): Port number for the FastAPI server. Default is 5000.

        Returns:
            str: Path to the generated FastAPI script.
        """
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting API generation for model: {model.get_model_name()}")
        
        try:
            x = pd.DataFrame(self._setup.X_train)
            y = pd.DataFrame(self._setup.y_train)
            model_name = model.get_model_name()
            logger.info(f"Training data obtained. X shape: {x.shape}, y shape: {y.shape}")

            print(api_name)
            if api_name == "":
                api_name = f"{self.filename}_{model_name}"
            logger.info(f"API name: {api_name}")

            score = 0.8
            input_data = x.iloc[0].to_dict()
            output_data = {"prediction": y.iloc[0][0], "score": score}
            model_result = "model['trained_model']"

            dataset_features = {}
            
            dataset_columns = self._dataset.columns
            logger.info(f"Dataset columns: {list(dataset_columns)}")

            for column in dataset_columns:
                min_val = self._dataset[column].min()
                max_val = self._dataset[column].max()
                dataset_features[column] = (min_val, max_val)
            logger.info(f"Dataset features processed: {len(dataset_features)} columns")
        except Exception as e:
            logger.error(f"Error processing data for API generation: {str(e)}")
            raise e
        print(dataset_features)
        
        dataset_columns_str = ','.join(f'"{str(item)}"' for item in dataset_columns)
        api_name_f = api_name.split("/")[-1]
        query = f"""# -*- coding: utf-8 -*-
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import create_model
from os.path import exists
from scipy.stats import ks_2samp
import os

app = FastAPI()
cwd = os.getcwd()

model = joblib.load("./tmp/files/{api_name_f}.pkl")

tmp_dataset = cwd + "/tmp/files/{api_name_f}.csv"

if os.path.exists(tmp_dataset):
    df_tmp = pd.read_csv(tmp_dataset)
else:
    df_tmp = pd.DataFrame(columns=[{dataset_columns_str}])
    df_tmp.to_csv(tmp_dataset, index=False)

input_model = create_model("{api_name_f}_input", **{input_data})
output_model = create_model("{api_name_f}_output", **{output_data})

def check_uploaded_file(file: File(...)):
    try:
        filename = file.filename.split("/")[-1].split('.')[0]
        contents = file.file.read()
        if file.content_type == "text/csv":
            file_path = cwd+"/files/uploads/"+file.filename
            with open(file_path, 'wb') as f:
                f.write(contents)
            print("file_read")
        else:
            return None, {{'message': 'Incorrect file type. Only CSV files are accepted'}}
    except Exception:
        return None, {{'message': 'Error uploading file'}}
    finally:
        file.file.close()
    message = 'File saved at ' + file_path
    return file_path, {{"message": message}}

def check_drift_ks(new_data, old_data, alpha=0.05):
    old_predictions = model.predict(old_data)
    new_predictions = model.predict(new_data)
    statistic, p_value = ks_2samp(old_predictions, new_predictions)
    drift_detected = p_value < alpha
    return drift_detected

@app.get("/")
def check_name():
    return {{"model": "{model.get_model_name()}"}}

@app.get("/get_features")
def get_features():
    return {dataset_features}

@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]
    score = probabilities[prediction]
    data["{self._dataset.columns[self._target_position]}"] = prediction
    print(data)
    global df_tmp
    df_tmp = pd.concat([df_tmp, pd.DataFrame(data)])
    df_tmp.to_csv(tmp_dataset, index=False)
    return {{"prediction": prediction, "score":score}}

@app.post("/check_drift")
def check_drift(new_data: UploadFile = File(...), old_data: UploadFile = File(...)):
    new_file, message = check_uploaded_file(new_data)
    if new_file is None:
        return message
    old_file, message = check_uploaded_file(old_data)
    if old_file is None:
        return message
    new_data = pd.read_csv('../Uploads/dataset_edit.csv')
    X_new = new_data.iloc[:,:-1]
    y_new = new_data.iloc[:,-1]
    old_data = pd.read_csv('../Uploads/dataset_edit.csv')
    X_old = old_data.iloc[:,:-1]
    y_old = old_data.iloc[:,-1]
    x = check_drift_ks(X_new, X_old, alpha=2)
    res = "False" if x == False else "True"
    return {{"check":  res}}

if __name__ == "__main__":
    uvicorn.run("{api_name_f}:app", host="{host}", port={port})
"""

        path_to_save = f"{api_name}.py"
        logger.info(f"Saving API file at: {path_to_save}")

        try:
            with open(path_to_save, "w") as file:
                file.write(query)
            logger.info(f"API file saved successfully: {path_to_save}")
        except Exception as e:
            logger.error(f"Error saving API file: {str(e)}")
            raise e

        print(
            "API created successfully. This function only creates a POST API, "
            "it is not executed automatically. To run your API, please execute this "
            f"command --> !python {os.getcwd()}/{api_name}.py"
        )

        result_path = f"python {os.path.expanduser(os.getcwd())}/{api_name}.py"
        logger.info(f"Returning API path: {result_path}")
        return result_path