# -*- coding: utf-8 -*-


# standard library imports
from contextlib import nullcontext
import os
import getpass
from pathlib import Path
import pickle
import sys
import json
import logging


# third party imports
# import mlflow


# local application imports
from func.setup.FIGConnect import FIGConnect
from func.utils import setup_logging, get_git_commit_hash


class Setup:
    """ This class handles FIGConnect setups and loggings. It also saves artifacts."""


    __allowed = {"PROJECT_NAME":str, "RUN_ID":str, "SAVE_ARTIFACTS":bool, "LOGGING":bool,
                 "LOGGING_MODE":str|None, "ROLE":str, "NETWORK_ID":str, "PROJECT_DIR":str,
                 "RUN_FOLDER":str, "MLFLOW_TRACKING":bool, "MLFLOW_EXPERIMENT_NAME":str,
                 'MEMORY_REDUCTION':bool}
    __folders = ("a_business_understanding", "b_data_acquisition", "c_modeling", "d_deployment", "e_customer_acceptance", "f_communication")


    def __init__(self, params: dict = {}, **kwargs):
        """
        Initialize the project setup, database connections, and create a run directory, with logging and artifact saving options.


        Args:
            PROJECT_NAME (str): Name of the project.
            RUN_ID (str): Unique identifier for the run. Default is current timestamp.
            SAVE_ARTIFACTS (bool): Save artifacts to the run directory. Default is False.
            LOGGING (bool): Redirect log to a log file. Default is False.
            LOGGING_MODE (str): 'a' for append, 'w' for write. Default is 'a'.
            MEMORY_REDUCTION (bool): Reduce memory usage. Default is False.
            ROLE (str): Snowflake role override; otherwise use environment or connection profile.
            NETWORK_ID (str): User's network ID. Default is getpass.getuser().lower(). Not used yet.
            PROJECT_DIR (str): Project directory. Default is the directory of this file.
            RUN_FOLDER (str): Directory to save runs under PROJECT_DIR. Default is "runs".
            ARTIFACT_PATH (str): Path to save artifacts. Default is PROJECT_DIR/RUN_FOLDER/RUN_ID.
            MLFLOW_TRACKING (bool): Use MLFlow tracking. Default is False. Not used yet.
            MLFLOW_EXPERIMENT_NAME (str): Name of the MLFlow experiment. Default is "default". Not used yet.


        Returns:
            None
        """
        import datetime


        # set default parameters
        self.PROJECT_NAME = "Home_Retention_13_1"
        self.SAVE_ARTIFACTS = False
        self.LOGGING = False
        self.LOGGING_MODE = None            # a for append, w for write
        self.MEMORY_REDUCTION = False
        self.ROLE = None
        self.NETWORK_ID = getpass.getuser().lower()     # getpass.getuser()
        self.PROJECT_DIR = Path(os.path.dirname(__file__)).resolve()
        self.RUN_FOLDER = "runs"
        self.RUN_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')
        self.ARTIFACT_PATH = None
        self.MLFLOW_TRACKING = False
        self.MLFLOW_EXPERIMENT_NAME = "default"


        # update parameters
        for k, v in params.items():
            assert k in self.__class__.__allowed, f"Invalid parameter: {k}. Allowed parameters are {list(self.__allowed.keys())}"
            assert isinstance(v, self.__class__.__allowed[k]), f"Invalid type for parameter: {k}. Expected {self.__class__.__allowed[k]}, got {type(v)}"
            setattr(self, k, v)


        for k, v in kwargs.items():
            assert k in self.__class__.__allowed, f"Invalid parameter: {k}. Allowed parameters are {list(self.__allowed.keys())}"
            assert isinstance(v, self.__class__.__allowed[k]), f"Invalid type for parameter: {k}. Expected {self.__class__.__allowed[k]}, got {type(v)}"
            setattr(self, k, v)


        # update ARTIFACT_PATH in case it is given or PROJECT_DIR, RUN_FOLDER, RUN_ID gets updated
        self.ARTIFACT_PATH = self.ARTIFACT_PATH or Path(os.path.join(self.PROJECT_DIR, self.RUN_FOLDER, self.RUN_ID))
        self.params = self.__dict__


        if self.ARTIFACT_PATH.exists():
            print(f"Run directory {self.ARTIFACT_PATH} already exists. Try loading artifacts.")
            # load the artifact if it exists
            if os.path.exists(os.path.join(self.ARTIFACT_PATH, "artifacts.pkl")):
                self.artifacts = pickle.load(open(os.path.join(self.ARTIFACT_PATH, "artifacts.pkl"), "rb"))
                print("Artifacts loaded")
            else:
                self.artifacts = {i: {} for i in self.__folders}
            if self.LOGGING_MODE == 'w':
                print("Logging mode is set to overwrite. Change to self.LOGGING_MODE = 'a' (append) and run .set_logging() to avoid overwriting the log file.")
        else:
            # create the artifact template
            self.artifacts = {i: {} for i in self.__folders}


        # create a run path if LOGGING or SAVE_ARTIFACTS is True
        if self.LOGGING or self.SAVE_ARTIFACTS:
            self._make_artifact_dir()


        # Change the project name only for cookiecutter template development.
        if "cookiecutter.project_slug" in self.PROJECT_NAME:
            self.PROJECT_NAME = "cookiecutter"
            print("Default PROJECT_NAME: " + self.PROJECT_NAME)


        # redirect print output from console to a file
        self.logger = logging.getLogger(__name__)
        if self.LOGGING:
            # self.original_stdout = sys.stdout
            # self.f = open(os.path.join(self.params["ARTIFACT_PATH"], "log.txt"), "w")
            # sys.stdout = self.f
            self.LOGGING_MODE = self.LOGGING_MODE or 'a'
            self.set_logging()
        _branch, _commit = get_git_commit_hash()
        self.logger.info(f"Git branch: {_branch}. Commit hash: {_commit}.")


        self.conns = build_conn(self.ROLE, memory_reduction=self.MEMORY_REDUCTION)


        # TODO: setup MLFlow Tracking
        # if self.params["MLFLOW_TRACKING"]:
        #     print("Initialize MLFlow Tracking")
        #     print("Tracking URI: http://localhost:5000")
        #     mlflow.set_tracking_uri("http://localhost:5000")
        #     mlflow.set_experiment(self.params["MLFLOW_EXPERIMENT_NAME"])
        #     mlflow.autolog()
        #     # trying to find the current artifact directory to debug could not find a registered artifact repository error
        #     artifact_uri =  mlflow.get_artifact_uri()
        #     print(artifact_uri)
        #     try:
        #         self.mlflow_context = mlflow.start_run(run_name = run_name)
        #     except:
        #         mlflow.end_run()
        #         self.mlflow_context = mlflow.start_run(run_name = run_name)
        # else:
        #     print("No MLFlow Tracking")
        #     self.mlflow_context = nullcontext()


        # build connection objects
        # Connection wrappers remain lazy until an explicit warehouse query.


    def _make_artifact_dir(self):
        Path(self.ARTIFACT_PATH).mkdir(parents=True, exist_ok=True)


    def get_dir(self, prefix: str = None):
        if not prefix:
            return [os.path.join(self.PROJECT_DIR, i) for i in self.__folders]
        return os.path.join(self.PROJECT_DIR, [i for i in self.__folders if i.startswith(prefix.lower())][0])


    def set_logging(self):
        """ Set up logging for the run. Allows users to set it up later if they did not intend to log initially."""
        if not self.LOGGING:
            res = input("You did not intend to log. Do you want to log now? (y/n): ")
            assert res.lower() in ('y', 'n'), "Invalid response. Please enter 'y' or 'n'"
            if res.lower() == 'n':
                print("Logging not set")
                return
            self.LOGGING = True
            res = input("Enter logging mode (a for append, [w] for write, n to exit): ")
            assert res.lower() in ('a', 'w', 'n'), "Invalid response. Please enter 'a' or 'w' or 'n'"
            if res.lower() == 'n':
                print("Logging not set")
                return
            self.LOGGING_MODE = res.lower()
            self._make_artifact_dir()
        setup_logging(os.path.join(self.ARTIFACT_PATH, "log.log"), self.LOGGING_MODE, os.path.join(Path(os.path.dirname(__file__)).resolve(), "logconfig.json"))




    def set_artifacts(self):
        """ Save artifacts to the run directory. Allows users to save artifacts if they did not intend to initially."""
        if not self.SAVE_ARTIFACTS:
            # check if want to save anyway
            res = input("You did not intend to save artifacts. Do you want to save the artifacts now? (y/n): ")
            assert res.lower() in ('y', 'n'), "Invalid response. Please enter 'y' or 'n'"
            if res.lower() == 'n':
                print("Artifacts not saved")
                return
            self.SAVE_ARTIFACTS = True
            self._make_artifact_dir()


        self.artifacts["PARAMS"] = {k: v for k, v in self.params.items() if k in self.__class__.__allowed}
        # save query info, other connection queries would be appended
        self.artifacts["b_data_acquisition"]["queries"] = {conn_name: conn.queries for conn_name, conn in self.conns.items() if conn.queries}
        file_name = os.path.join(self.ARTIFACT_PATH, "artifacts.pkl")
        with open(file_name, 'wb') as f:
            res = pickle.dump(self.artifacts, f)
        if self.MLFLOW_TRACKING:
            pass    # TODO
            # mlflow.log_artifacts(self.params['ARTIFACT_PATH'])
        print(f"Artifacts saved to {file_name}")


    def save_config(self, config: dict, filename: str = 'config.json', mode: str = 'w', logging : bool = True):
        with open(f'{self.ARTIFACT_PATH}/{filename}', mode) as file:
            json.dump(config, file, indent=4)
        if logging:
            self.logger.info(f'config: \n{json.dumps(config, indent=4)}')


    def load_config(self, filename: str = 'config.json', logging : bool = True) -> dict:
        with open(f'{self.ARTIFACT_PATH}/{filename}', 'r') as file:
            config = json.load(file)
        if logging:
            self.logger.info(f'config: \n{json.dumps(config, indent=4)}')
        return config


    def update_config(self, config: dict, filename: str = 'config.json', logging : bool = True):
        _config = self.load_config(filename, logging=False)
        _config.update(config)
        self.save_config(_config, filename, 'w', logging=False)
        if logging:
            self.logger.info(f'config: \n{json.dumps(config, indent=4)}')




    def close_conns(self, conns: dict = None):
        """
        Close all db connections


        Parameters
        ----------
        conns : dict
            A dictionary of FIGConnect connection objects. Typically setup.conns


        Returns
        -------
        None.


        """
        if conns is None:
            conns = self.conns
        for i, (key, value) in enumerate(conns.items()):
            verbose = 20 if i == 0 else 0
            value.close_conn(verbose=verbose)


    def shutdown(self):
        # if self.params["LOGGING"]:
        #     sys.stdout = self.original_stdout
        #     self.f.close()
        if self.LOGGING:
            logging.shutdown()


        if self.SAVE_ARTIFACTS:
            self.set_artifacts()


        self.close_conns()


def build_conn(role: str = None, memory_reduction: bool = False) -> dict:
    """Construct the retention workflow's lazy public connection wrappers.

    Database/schema values match params and the build SQL renderer. Neither
    configuration validation nor authentication occurs until an explicit query.
    """
    from params import DATABASE, SCHEMA, SF_CONN, SP_CONN
    common = {"database": DATABASE, "schema": SCHEMA,
              "memory_reduction": memory_reduction}
    if role:
        common["role"] = role
    return {
        SF_CONN: FIGConnect(server="SNOWFLAKE", **common),
        SP_CONN: FIGConnect(server="SNOWPARK", **common),
    }


def clean_config(config: dict, exclude: list = []):
    """
    Clean the configuration dictionary by removing keys that are in the exclude list.


    Parameters:


        config: The configuration dictionary to clean.


        exclude: A list of keys to exclude from the configuration.


    Returns:
        dict: A cleaned configuration dictionary with specified keys removed.
    """
    return {k:v for k,v in config.items() \
            if k not in exclude and not k.startswith('_') and not k.startswith('.') and k not in ['globals', 'locals']}

