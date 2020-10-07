import click
import mlflow

from myautoml.utils.mlflow import track_model_from_file, track_model_data, register_model


@click.command()
@click.option('--experiment_name', '-e', help='Experiment name in MLflow')
@click.option('--model_path', '-m', help='Path to the trained model file')
@click.option('--model_name', '-n', help='Name to register the model with')
def registermodel(experiment_name, model_path, model_name):
    if experiment_name is None:
        experiment_name = "Default"
    if model_path is not None:
        click.secho("Uploading model to MLflow Server", fg="green", bold=True)
        run_info = track_model_from_file(local_path=model_path,
                                         experiment_name=experiment_name)
        track_model_data(run_info.run_id, tags={"cli_upload": True})
        experiment_run_url = f"{mlflow.get_tracking_uri()}/#/experiments/{run_info.experiment_id}/runs/{run_info.run_id}"

        if model_name is not None:
            click.secho("Registering model with MLflow Server", fg="green", bold=True)
            model_version = register_model(run_id=run_info.run_id,
                                           model_name=model_name)
            registered_model_url = f"{mlflow.get_tracking_uri()}/#/models/{model_name}/versions/{model_version.version}"
        else:
            click.secho(f"No name specified to register the model!", fg="white", bold=True)
            registered_model_url = ""

        click.echo(f"Experiment run URL: {experiment_run_url}")
        click.echo(f"Registered model URL: {registered_model_url}")

    else:
        click.secho(f"No model specified to upload!", fg="red", bold=True)
