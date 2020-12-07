import logging

import click
from dotenv import load_dotenv
import mlflow

from myautoml.utils.mlflow.tracking import track_sk_model_from_file

_logger = logging.getLogger(__name__)


@click.command()
@click.option('--experiment_name', '-e', help='Experiment name in MLflow')
@click.option('--model_path', '-m', help='Path to the trained model file')
@click.option('--model_name', '-n', help='Name to register the model with')
@click.option('--log_level', help='Logging level of messages to log to console', default='INFO')
@click.option('--dotenv', help='Path to the dotenv file with environment variables')
def registermodel(experiment_name, model_path, model_name, log_level, dotenv):
    if dotenv:
        load_dotenv(dotenv)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=log_level
    )

    logging.getLogger('azure').setLevel('WARNING')
    logging.getLogger('git').setLevel('INFO')
    logging.getLogger('matplotlib').setLevel('INFO')
    logging.getLogger('urllib3').setLevel('INFO')

    if experiment_name is None:
        experiment_name = "Default"
    if model_path is not None:
        run_info = track_sk_model_from_file(
            local_path=model_path,
            experiment_name=experiment_name,
            run_name=None,
            model_artifact_path='model',
            registered_model_name=model_name,
            params=None,
            tags={"cli_upload": True},
            metrics=None,
            artifacts=None
        )
        experiment_run_url = f"{mlflow.get_tracking_uri()}/#/experiments/{run_info.experiment_id}/runs/{run_info.run_id}"
        click.echo(f"Experiment run URL: {experiment_run_url}")
        # registered_model_url = f"{mlflow.get_tracking_uri()}/#/models/{model_name}/versions/{model_version.version}"
        # click.echo(f"Registered model URL: {registered_model_url}")

    else:
        click.secho(f"No model specified to upload!", fg="red", bold=True)
