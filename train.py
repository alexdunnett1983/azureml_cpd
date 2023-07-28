import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from sklearn.ensemble import RandomForestClassifier

#authenticate
credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id="c6fb9fd9-644b-44c5-8e1f-2ea146326c95",
    resource_group_name="Alexander.Dunnett-rg",
    workspace_name="demo-alexplore"
)
df = pd.read_parquet(ml_client.data.get(name="credit_card",version="2023.07.06.22.03.34_cleaned").path)
X_train, X_test, y_train, y_test = train_test_split(df.drop('default',axis=1), df['default'], test_size=0.25, random_state=42)

#using MLflow to track development: w/ autologging
mlflow.set_experiment("Training credit-card fraud")
mlflow.sklearn.autolog()

#run classifier
mlflow.start_run()
rnd = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
rnd.fit(X_train,y_train)
y_pred = rnd.predict(X_test)
classification_report(y_test,y_pred)
mlflow.end_run()