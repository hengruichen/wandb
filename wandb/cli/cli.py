#!/usr/bin/env python

import asyncio
import configparser
import datetime
import getpass
import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from functools import wraps
from typing import Any, Dict, Optional

import click
import yaml
from click.exceptions import ClickException

# pycreds has a find_executable that works in windows
from dockerpycreds.utils import find_executable

import wandb
import wandb.env

# from wandb.old.core import wandb_dir
import wandb.sdk.verify.verify as wandb_verify
from wandb import Config, Error, env, util, wandb_agent, wandb_sdk
from wandb.apis import InternalApi, PublicApi
from wandb.apis.public import RunQueue
from wandb.integration.magic import magic_install
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.launch import utils as launch_utils
from wandb.sdk.launch._launch_add import _launch_add
from wandb.sdk.launch.errors import ExecutionError, LaunchError
from wandb.sdk.launch.sweeps import utils as sweep_utils
from wandb.sdk.launch.sweeps.scheduler import Scheduler
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.wburls import wburls
from wandb.sync import TMPDIR, SyncManager, get_run_from_path, get_runs

# Send cli logs to wandb/debug-cli.<username>.log by default and fallback to a temp dir.
_wandb_dir = wandb.old.core.wandb_dir(env.get_dir())
if not os.path.exists(_wandb_dir):
    _wandb_dir = tempfile.gettempdir()

try:
    _username = getpass.getuser()
except KeyError:
    # getuser() could raise KeyError in restricted environments like
    # chroot jails or docker containers. Return user id in these cases.
    _username = str(os.getuid())

_wandb_log_path = os.path.join(_wandb_dir, f"debug-cli.{_username}.log")

logging.basicConfig(
    filename=_wandb_log_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("wandb")

# Click Contexts
CONTEXT = {"default_map": {}}
RUN_CONTEXT = {
    "default_map": {},
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}


def cli_unsupported(argument):
    wandb.termerror(f"Unsupported argument `{argument}`")
    sys.exit(1)


class ClickWandbException(ClickException):
    def format_message(self):
        # log_file = util.get_log_file_path()
        log_file = ""
        orig_type = f"{self.orig_type.__module__}.{self.orig_type.__name__}"
        if issubclass(self.orig_type, Error):
            return click.style(str(self.message), fg="red")
        else:
            return (
                f"An Exception was raised, see {log_file} for full traceback.\n"
                f"{orig_type}: {self.message}"
            )


def display_error(func):
    """Function decorator for catching common errors and re-raising as wandb.Error."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except wandb.Error as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            logger.error("".join(lines))
            wandb.termerror(f"Find detailed error logs at: {_wandb_log_path}")
            click_exc = ClickWandbException(e)
            click_exc.orig_type = exc_type
            raise click_exc.with_traceback(sys.exc_info()[2])

    return wrapper


_api = None  # caching api instance allows patching from unit tests


def _get_cling_api(reset=None):
    """Get a reference to the internal api with cling settings."""
    global _api
    if reset:
        _api = None
        wandb_sdk.wandb_setup._setup(_reset=True)
    if _api is None:
        # TODO(jhr): make a settings object that is better for non runs.
        # only override the necessary setting
        wandb.setup(settings
# ... [truncated] ...
 service",
)
def disabled(service):
    api = InternalApi()
    try:
        api.set_setting("mode", "disabled", persist=True)
        click.echo("W&B disabled.")
        os.environ[wandb.env._DISABLE_SERVICE] = str(service)
    except configparser.Error:
        click.echo(
            "Unable to write config, copy and paste the following in your terminal to turn off W&B:\nexport WANDB_MODE=disabled"
        )


@cli.command("enabled", help="Enable W&B.")
@click.option(
    "--service",
    is_flag=True,
    show_default=True,
    default=True,
    help="Enable W&B service",
)
def enabled(service):
    api = InternalApi()
    try:
        api.set_setting("mode", "online", persist=True)
        click.echo("W&B enabled.")
        os.environ[wandb.env._DISABLE_SERVICE] = str(not service)
    except configparser.Error:
        click.echo(
            "Unable to write config, copy and paste the following in your terminal to turn on W&B:\nexport WANDB_MODE=online"
        )


@cli.command("gc", hidden=True, context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1)
def gc(args):
    click.echo(
        "`wandb gc` command has been removed. Use `wandb sync --clean` to clean up synced runs."
    )


@cli.command(context_settings=CONTEXT, help="Verify your local instance")
@click.option("--host", default=None, help="Test a specific instance of W&B")
def verify(host):
    # TODO: (kdg) Build this all into a WandbVerify object, and clean this up.
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_PROJECT"] = "verify"
    api = _get_cling_api()
    reinit = False
    if host is None:
        host = api.settings("base_url")
        print(f"Default host selected: {host}")
    # if the given host does not match the default host, re-run init
    elif host != api.settings("base_url"):
        reinit = True

    tmp_dir = tempfile.mkdtemp()
    print(
        "Find detailed logs for this test at: {}".format(os.path.join(tmp_dir, "wandb"))
    )
    os.chdir(tmp_dir)
    os.environ["WANDB_BASE_URL"] = host
    wandb.login(host=host)
    if reinit:
        api = _get_cling_api(reset=True)
    if not wandb_verify.check_host(host):
        sys.exit(1)
    if not wandb_verify.check_logged_in(api, host):
        sys.exit(1)
    url_success, url = wandb_verify.check_graphql_put(api, host)
    large_post_success = wandb_verify.check_large_post()
    wandb_verify.check_secure_requests(
        api.settings("base_url"),
        "Checking requests to base url",
        "Connections are not made over https. SSL required for secure communications.",
    )
    if url:
        wandb_verify.check_secure_requests(
            url,
            "Checking requests made over signed URLs",
            "Signed URL requests not made over https. SSL is required for secure communications.",
        )
        wandb_verify.check_cors_configuration(url, host)
    wandb_verify.check_wandb_version(api)
    check_run_success = wandb_verify.check_run(api)
    check_artifacts_success = wandb_verify.check_artifacts()
    if not (
        check_artifacts_success
        and check_run_success
        and large_post_success
        and url_success
    ):
        sys.exit(1)


@cli.group("import", help="Commands for importing data from other systems")
def importer():
    pass


@importer.command("mlflow", help="Import from MLFlow")
@click.option("--mlflow-tracking-uri", help="MLFlow Tracking URI")
@click.option(
    "--target-entity", required=True, help="Override default entity to import data into"
)
@click.option(
    "--target-project",
    required=True,
    help="Override default project to import data into",
)
def mlflow(mlflow_tracking_uri, target_entity, target_project):
    from wandb.apis.importers import MlflowImporter

    importer = MlflowImporter(mlflow_tracking_uri=mlflow_tracking_uri)
    overrides = {
        "entity": target_entity,
        "project": target_project,
    }

    importer.import_all_parallel(overrides=overrides)

