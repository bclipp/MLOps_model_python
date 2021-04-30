import math as math

import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import mlflow
import mlflow.sklearn
import sys
from pyspark.sql import SparkSession


def main():
    spark = SparkSession \
        .builder \
        .appName("MLOps_model_python") \
        .getOrCreate()
    uid = sys.argv[1]
    spark.conf.set("spark.sql.execution.arrow.enabled", True)
    print(f"reading delta table: dbfs:/datalake/stocks_{uid}/data")
    try:
        df = spark.read.format("delta").load(f"dbfs:/datalake/stocks_{uid}/data")
    except Exception as e:
        print(f"There was an error loading the delta stock table, : error:{e}")
    pdf = df.select("*").toPandas()
    df_2 = pdf.loc[:, ["AdjClose", "Volume"]]
    df_2["High_Low_Pert"] = (pdf["High"] - pdf["Low"]) / pdf["Close"] * 100.0
    df_2["Pert_change"] = (pdf["Close"] - pdf["Open"]) / pdf["Open"] * 100.0
    df_2.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.01 * len(df_2)))
    forecast_col = "AdjClose"
    df_2['label'] = df_2[forecast_col].shift(-forecast_out)
    X = np.array(df_2.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_2['label'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print("creating MLflow project")
    experiment_id = mlflow.create_experiment(f"/Users/bclipp770@yandex.com/datalake/stocks/experiments/{uid}")
    experiment = mlflow.get_experiment(experiment_id)
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("building our model")
    with mlflow.start_run():
        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        mae = metrics.mean_absolute_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        mlflow.log_param("max_depth", 2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(regr, "model")
        model_path = f"dbfs:/datalake/stocks_{uid}/model"
        mlflow.sklearn.save_model(regr, model_path)


if __name__ == "__main__":
    main()
