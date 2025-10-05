#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow
from prefect import task, flow, get_run_logger
from prefect.artifacts import create_table_artifact, create_markdown_artifact
from sklearn.ensemble import RandomForestClassifier

from etl import GetData
from feature_engineer import FeatureEngineer
from train_with_mlflow_optuna import TrainMlflowOptuna

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow configuration with fallback
def setup_mlflow():
    """Setup MLflow with proper error handling and fallback options."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///wine_quality.db")
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        # Test connection
        mlflow.search_experiments()
        print(f"MLflow tracking URI set to: {mlflow_uri}")
        logger.info(f"Connected to MLflow at: {mlflow_uri}")
    except Exception as e:
        logger.warning(f"Failed to connect to {mlflow_uri}: {e}")
        logger.info("Falling back to local SQLite database")
        mlflow.set_tracking_uri("sqlite:///wine_quality.db")
    
    try:
        mlflow.set_experiment("wine-quality-experiment-prefect")
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        raise

# Initialize MLflow
#setup_mlflow()


@task(name="load_data", description="Carga del set de datos de la calidad de vinos desde Kaggele", retries=3, retry_delay_seconds=10)
def read_dataframe() -> pd.DataFrame:
    """
    Carga del set de datos de la calidad de vinos desde Kaggele

    Args:
        None

    Returns:
        Dataframe procesado
    """


    logger = get_run_logger()

    get_data = GetData()
    logger.info(f"Cargando el set de datos de Kaggle: {get_data.kaggle_dataset_id}")

    get_data.download_data()
    
    try:
        df = get_data.create_dataset()
        logger.info(f"Carga satisfactoria, {len(df)} registros cargados.")
    except Exception as e:
        logger.error(f"Error al cargar los datos: {get_data.kaggle_dataset_id}: {e}")
        raise

    # Create artifact with data summary
    summary_data = [
        ["Total Records", len(df)],
        #["Average Duration", f"{df['duration'].mean():.2f} minutes"],
        #["Min Duration", f"{df['duration'].min():.2f} minutes"],
        #["Max Duration", f"{df['duration'].max():.2f} minutes"],
        #["Unique PU_DO combinations", df['PU_DO'].nunique()]
    ]

    create_table_artifact(
        key=f"data-summary",
        table=summary_data,
        description=f"Data summary for wine quality dataset (n={len(df)})"
    )

    return df


@task(name="create_features", description="Create feature matrix using DictVectorizer")
def create_features(df: pd.DataFrame) -> Tuple[any, list, list, str]:
    """
    Create feature matrix from DataFrame.

    Args:
        df: Input DataFrame
        dv: Pre-fitted DictVectorizer (optional)

    Returns:
        Tuple of (feature matrix, DictVectorizer)
    """
    logger = get_run_logger()

    # Feature engineering
    feature_engineer = FeatureEngineer(df)
    df = feature_engineer.create_features()

    # Variables categoricas, numéricas y variable objetivo
    numeric_features = ['volatile_acidity', 'residual_sugar', 'density', 'alcohol']
    categorical_features = []
    target_column = 'quality'

    return df, numeric_features, categorical_features, target_column


@task(name="train_model", description="Train RandomForestClassifier model with MLflow tracking")
def train_model(df, numeric_features, categorical_features, target_column) -> str:
    """
    Train RandomForestClassifier model and log to MLflow.

    Args:
        df: Input DataFrame
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        target_column: Name of the target column

    Returns:
        MLflow run ID
    """
    logger = get_run_logger()
    
    # Ensure models directory exists
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)
    
    logger.info(f"Training with {df.shape[0]} samples, {df.shape[1] - 1} features")

    param_distributions = {
        'n_estimators': ('int', 50, 200),
        'max_depth': ('int', 5, 30),
        'min_samples_split': ('int', 2, 10),
        'min_samples_leaf': ('int', 1, 5),
        'max_features': ('categorical', ['sqrt', 'log2', None])
    }

    trainer = TrainMlflowOptuna(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=target_column,
        model_class=RandomForestClassifier,
        test_size=0.3,
        n_trials=30,
        optimization_metric='f1',  # Optimize for F1 score
        param_distributions=param_distributions,
        model_params={'random_state': 42, 'n_jobs': -1},
        mlflow_setup = mlflow
    )

    best_pipeline, run_id, study = trainer.train()


    logger.info(f"\nOptimization complete!")
    logger.info(f"Best f1: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    # Create Prefect artifact with model performance
    performance_data = [
        ["f1", f"{study.best_value:.4f}"],
        ["Max Depth", study.best_params['max_depth']],
        ["min_samples_split", study.best_params['min_samples_split']],
        ["min_samples_leaf", study.best_params['min_samples_leaf']],
        ["max_features", study.best_params['max_features']],
        ["MLflow Run ID", run_id]
    ]

    create_table_artifact(
            key="model-performance",
            table=performance_data,
            description=f"Model performance metrics - f1: {study.best_value:.4f}"
        )
    
    # Create markdown artifact with training summary
    markdown_content = f"""
    # Model Training Summary

    ## Performance
    - **f1**: {study.best_value:.4f}
    - **MLflow Run ID**: {run_id}

    ## Parameters
    - Max Depth: {study.best_params['max_depth']}
    - min_samples_split: {study.best_params['min_samples_split']}
    - min_samples_leaf: {study.best_params['min_samples_leaf']}
    - max_features: {study.best_params['max_features']}

    ## Training Details
    - Boost Rounds: 30
    - Early Stopping: 50 rounds
    - Best pipeline: {best_pipeline}
    """

    create_markdown_artifact(
            key="training-summary",
            markdown=markdown_content,
            description="Detailed training summary"
        )

    logger.info(f"Training complete!")

    return run_id


@flow(name="Wine Quality Prediction Pipeline", description="End-to-end ML pipeline for wine quality prediction")
def wine_quality_prediction_flow() -> str:
    """
    Main flow for wine quality prediction.

    Args:
        None

    Returns:
        MLflow run ID
    """
    # Load training data
    df = read_dataframe()

    # Create features
    df_train, numeric_features, categorical_features, target_column = create_features(df)

    # Train model
    run_id = train_model(df_train, numeric_features, categorical_features, target_column)

    # Create final pipeline artifact
    pipeline_summary = f"""
    # Pipeline Execution Summary

    ## Data
    - **Training Period**: https://www.kaggle.com/datasets/rajyellow46/wine-quality

    ## Results
    - **MLflow Run ID**: {run_id}
    - **MLflow Experiment**: wine-quality-experiment-prefect

    ## Next Steps
    1. Review model performance in MLflow UI: http://localhost:5000
    2. Compare with previous runs
    3. Consider model deployment if performance is satisfactory
    """

    create_markdown_artifact(
        key="pipeline-summary",
        markdown=pipeline_summary,
        description="Complete pipeline execution summary"
    )

    return run_id


if __name__ == "__main__":
    #import argparse

    #parser = argparse.ArgumentParser(description='Entrenar modelo para predecir la calidad del vino') 
    #parser.add_argument('--mlflow-uri', type=str, help='MLflow tracking URI (overrides environment variable)')
    #args = parser.parse_args()
    #print(f"argumentos: {args}")
    # Override MLflow URI if provided
    
    #os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///wine.db"
    setup_mlflow()

    try:
        # Run the flow
        run_id = wine_quality_prediction_flow()
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 MLflow run_id: {run_id}")
        print(f"🔗 View results at: {mlflow.get_tracking_uri()}")

        # Save run ID for reference
        with open("prefect_run_id.txt", "w") as f:
            f.write(run_id)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
