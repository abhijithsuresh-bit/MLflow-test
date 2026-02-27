import warnings
import os , sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import dagshub


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 1st we open github and create a new repo for the project , then open dagshub and click 'create' - > 'new repo' -> 'connect with github repo' -> select the repo we just created in github for this project,  'click' -> 'remote'-> 'experiments' entire url is there

    # --- DagsHub / MLflow setup (MUST be before start_run) ---
    dagshub.init(repo_owner="abhijithsuresh-bit", repo_name="MLflow-test", mlflow=True)

    remote_server_uri = "https://dagshub.com/abhijithsuresh-bit/MLflow-test.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)

    # Set or create an experiment (recommended for remote tracking)
    mlflow.set_experiment("wine-quality-elasticnet") # just a name for the experiment, you can choose any name you like

    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    print("Tracking store scheme:", urlparse(mlflow.get_tracking_uri()).scheme)

    # --- Data ---
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    data = pd.read_csv(csv_url, sep=";")

    train, test = train_test_split(data, test_size=0.25, random_state=42)

    train_x = train.drop(columns=["quality"])
    test_x = test.drop(columns=["quality"])

    # IMPORTANT: 1D targets for sklearn linear models
    train_y = train["quality"]
    test_y = test["quality"]

    # --- Params ---
    # when we do sys.argv, we get a list of command line arguments passed to the script. The first element (sys.argv[0]) is the script name, and the subsequent elements are the arguments. So, if we run python demo.py 0.1 0.2, then sys.argv will be ['demo.py', '0.1', '0.2']. Therefore, sys.argv[1] will be '0.1' and sys.argv[2] will be '0.2'. We convert these string values to float using float() function before using them as parameters for the ElasticNet model.
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5  # we need this code to change the parameters from the command line, otherwise we will use the default value of 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5 # we python demo.py 0.1 0.2 to change the parameters, otherwise we will use the default value of 0.5

    # --- Run ---
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE:  {mae}")
        print(f"  R2:   {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model as an artifact (works everywhere)
        mlflow.sklearn.log_model(lr, name="model")

