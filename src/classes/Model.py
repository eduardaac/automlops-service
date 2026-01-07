class Model:
    """Wrapper class for machine learning models.

    This class encapsulates a machine learning model provided as an estimator.
    It provides methods to retrieve information about the model, such as its name.

    Args:
        estimator: The machine learning model.

    Attributes:
        _estimator: The underlying machine learning estimator.

    Methods:
        __str__(): Returns a string representation of the model, equivalent to get_model_name().
        get_estimator(): Returns the underlying machine learning estimator.
        get_model_name(): Returns a human-readable name for the model.
    """
    def __init__(self, estimator) -> None:
        """Initialize the Model instance with the provided machine learning estimator."""
        self._estimator = estimator
        
    def __str__(self) -> str:
        """Return a string representation of the model, equivalent to get_model_name()."""
        return self.get_model_name()

    def get_estimator(self):
        """Return the underlying machine learning estimator."""
        return self._estimator

    def get_model_name(self) -> str:
        """Return a human-readable name for the model.

        Returns:
            str: The human-readable name of the model.
        """
        model_name = str(self._estimator).split("(")[0]
        if "catboost" in str(self._estimator):
            model_name = "CatBoostClassifier"
        model_dict_logging = {
            "ExtraTreesClassifier": "Extra_Trees_Classifier",
            "GradientBoostingClassifier": "Gradient_Boosting_Classifier",
            "RandomForestClassifier": "Random_Forest_Classifier",
            "LGBMClassifier": "Light_Gradient_Boosting_Machine",
            "XGBClassifier": "Extreme_Gradient_Boosting",
            "AdaBoostClassifier": "Ada_Boost_Classifier",
            "DecisionTreeClassifier": "Decision_Tree_Classifier",
            "RidgeClassifier": "Ridge_Classifier",
            "LogisticRegression": "Logistic_Regression",
            "KNeighborsClassifier": "K_Neighbors_Classifier",
            "GaussianNB": "Naive_Bayes",
            "SGDClassifier": "SVM_Linear_Kernel",
            "SVC": "SVM_Radial_Kernel",
            "GaussianProcessClassifier": "Gaussian_Process_Classifier",
            "MLPClassifier": "MLP_Classifier",
            "QuadraticDiscriminantAnalysis": "Quadratic_Discriminant_Analysis",
            "LinearDiscriminantAnalysis": "Linear_Discriminant_Analysis",
            "CatBoostClassifier": "CatBoost_Classifier",
            "BaggingClassifier": "Bagging_Classifier",
            "VotingClassifier": "Voting_Classifier",
        }
        result = model_dict_logging.get(model_name)
        if result is None:
            result = model_name.replace(" ", "_") if model_name else "Unknown_Model"
        return result