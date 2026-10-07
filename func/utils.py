from pathlib import Path
import os
import glob
import shutil
import pandas as pd
import logging.config
import json
from datetime import datetime
import pandas.io.formats.format as pf
import contextlib
import inspect
import re
from typing import Any
from itertools import islice
import logging
logger = logging.getLogger(__name__)


today = datetime.today().strftime('%Y%m%d')


def timer(func: callable) -> callable:
    """Timer decorator to measure the execution time of a function."""
    import time
    import functools


    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} ran in: {round((end - start) / 60, 2)} min')
        return result
    return wrapper


@timer
def load_sas_data(path: str, **kwargs) -> pd.DataFrame:
    """
    Load a SAS .sas7bdat file into a pandas DataFrame.


    Args:
        path: Path to the SAS .sas7bdat file.
        **kwargs: Additional keyword arguments passed to ``pd.read_sas()``.


    Returns:
        DataFrame containing the loaded SAS dataset.


    Notes:
        SAS stores dates as the number of days since 1960-01-01. Convert date
        columns after loading when needed, for example:


        SAS_EPOCH = pd.Timestamp('1960-01-01')
        date_col = 'your_date_column'
        if date_col in data.columns:
            data[date_col] = pd.to_timedelta(data[date_col], unit='D') + SAS_EPOCH
            logger.info(f'Converted {date_col} to datetime. Sample values:\n{data[date_col].head(5)}')
            logger.info(f'Max date: {data[date_col].max()}, Min date: {data[date_col].min()}')
        else:
            logger.warning(f'{date_col} column not found in the dataset!')
    """
    # utf-8 encoding not working, use the following instead
    # https://stackoverflow.com/questions/19699367/for-line-in-results-in-unicodedecodeerror-utf-8-codec-cant-decode-byte
    print('Loading SAS data...')
    data = pd.read_sas(path, format='sas7bdat', encoding='iso-8859-1', **kwargs)
    print('End loading')
    return data


def load_sas_data_with_pyreadstat(path: str, chunksize: int = None, **kwargs) -> pd.DataFrame:
    """
    Load SAS .sas7bdat data as a DataFrame.


    Args:
        path: path of the SAS .sas7bdat data
        chunksize: number of rows to read at a time
        **kwargs: additional keyword arguments to pass to pd.read_sas()
    Returns:
        A DataFrame if chunksize is None, else file reader which allows iteration
    """
    # utf-8 encoding not working, use the following instead
    # https://stackoverflow.com/questions/19699367/for-line-in-results-in-unicodedecodeerror-utf-8-codec-cant-decode-byte
    print('Loading SAS data...')
    import pyreadstat  # Optional SAS import; unrelated to Snowflake/model execution.
    data = pyreadstat.read_file_in_chunks(pyreadstat.read_sas7bdat, path, chunksize=chunksize, dates_as_pandas_datetime=True)
    print('End loading')
    return data


def _credential_prefix(service: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", str(service)).strip("_").upper()
    if not prefix:
        raise ValueError("Credential service name cannot be empty.")
    return prefix


def update_credentials(service: str, username: str, password: str, credential_path: str = None) -> None:
    """Set credentials in this process only; never persist secrets or use keyring.

    Prefer setting environment variables before launching the application.
    ``credential_path`` remains supported for explicit user-owned JSON inputs.
    """
    if credential_path:
        with open(credential_path, encoding="utf-8") as handle:
            values = json.load(handle)
        username, password = values["user"], values["password"]
    prefix = _credential_prefix(service)
    os.environ[f"{prefix}_USER"] = str(username)
    os.environ[f"{prefix}_PASSWORD"] = str(password)


def check_credentials(service: str, silent: bool = False):
    """Read explicit process credentials; no login prompt or keychain access.

    The Snowflake wrapper uses its richer authenticator/profile configuration
    directly. This compatibility helper is only for password-based callers.
    """
    prefix = _credential_prefix(service)
    username = os.getenv(f"{prefix}_USER")
    password = os.getenv(f"{prefix}_PASSWORD")
    if not username or not password:
        raise ValueError(f"Set {prefix}_USER and {prefix}_PASSWORD in the environment.")
    if not silent:
        return username, password


def setup_logging(log_path: str = 'log.log', mode: str = 'a', config_file: str = 'logconfig.json') -> None:
    """
    Setup logging configuration. Default logging levels are as follows.


        - CRITICAL: 50
        - FATAL: 50
        - ERROR: 40
        - WARN: 30
        - WARNING: 30
        - INFO: 20
        - DEBUG: 10
        - NOTSET: 0


    Args:
        log_path: path to the log file
        mode: file mode for logging, 'a' for append and 'w' for overwrite
        config_file: path to the logging config file in json format. The config file should follow
            the format specified in https://docs.python.org/3/library/logging.config.html#logging-config-dictschema.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    config['handlers']['file']['filename'] = log_path
    config['handlers']['file']['mode'] = mode
    logging.config.dictConfig(config)


def no_log() -> None:
    """
    Disable logging.
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    logging.config.dictConfig(config)


def check_logger(logger: logging.Logger, logging_level: int = logging.INFO) -> None:
    """Check if the logger has handlers. If not, add a stream handler to the logger.


    Args:
        logger: the logger to check
        logging_level: logging level to set if no handler is found. Default is logging.INFO.
    """
    import sys
    if not logger.hasHandlers():
        _ = logging.getLogger()
        if not _.hasHandlers():
            print('No handler found')
            # add stream handler if no handler is found
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.setLevel(logging_level)
        else:
            for handler in _.handlers:
                logger.addHandler(handler)


def convert_to_tuple(x: str|int|list|tuple) -> tuple:
    """Convert string, int, list to tuple. If input is already a tuple, return it.


    Args:
        x: input to convert, can be string, int, list, or tuple


    Returns:
        converted tuple
    """
    # convert string, int, list to tuple
    if isinstance(x, int):
        return f"({x})"
    elif isinstance(x, str):
        return f"('{x}')"
    elif isinstance(x, list):
        if len(x) == 1:
            return f"('{x[0]}')"
        return tuple(x)
    elif isinstance(x, tuple):
        return x
    else:
        raise ValueError("Invalid input type. Must be a string, list or tuple.")


def create_folder(path: str, folders: list, exist: str = 'skip') -> None:
    """
    Create folders in the path if not exists.


    Args:
        path: str, path to the folder
        folders: list, list of folders to create
        exist: str, 'skip' if skip if folder exists, 'overwrite' if overwrite the folder
    """
    assert exist in ['skip', 'overwrite'], "exist must be 'skip' or 'overwrite'."
    for f in folders:
        p = f'{path}/{f}'
        if not os.path.exists(p):
            os.makedirs(p)
            print(f'Folder {p} created.')
        elif exist == 'overwrite':
            shutil.rmtree(p)
            os.makedirs(p)
            print(f'Folder {p} overwritten.')
        elif exist == 'skip':
            os.makedirs(p, exist_ok=True)
            print(f'Folder {p} already exists. Skip.')


@contextlib.contextmanager
def custom_display(decimal: int = 2):
    """
    Custom display settings for pandas DataFrame
    Modified from https://stackoverflow.com/questions/29663252/can-you-format-pandas-integers-for-display-like-pd-options-display-float-format.


    Example:
        >>> with custom_display(decimal=2):
        ...     print(df)
    """
    orig_float_format = pd.options.display.float_format
    orig_max_rows = pd.options.display.max_rows
    orig_max_columns = pd.options.display.max_columns
    orig_width = pd.options.display.width
    orig_max_colwidth = pd.options.display.max_colwidth
    orig_expand_frame_repr = pd.options.display.expand_frame_repr
    orig_int_format = pf.IntArrayFormatter


    pd.options.display.float_format = f'{{:,.{decimal}f}}'.format
    pd.options.display.max_rows = 1000
    pd.options.display.max_columns = None
    pd.options.display.width = 1000
    pd.options.display.max_colwidth = 1000
    pd.options.display.expand_frame_repr = False


    class IntArrayFormatter(pf.GenericArrayFormatter):
        def _format_strings(self):
            formatter = self.formatter or '{:,d}'.format
            fmt_values = [formatter(x) for x in self.values]
            return fmt_values
    pf.IntArrayFormatter = IntArrayFormatter
    yield      # here is where the actual code will run
    # reset the settings
    pd.options.display.float_format = orig_float_format
    pd.options.display.max_rows = orig_max_rows
    pd.options.display.max_columns = orig_max_columns
    pd.options.display.width = orig_width
    pd.options.display.max_colwidth = orig_max_colwidth
    pd.options.display.expand_frame_repr = orig_expand_frame_repr
    pf.IntArrayFormatter = orig_int_format


def load_yaml(path: str) -> dict:
    """
    Load yaml file. This is developed for the purpose of loading config file, but it can be used for any yaml file.


    Args:
        path (str): path to the yaml file
    Returns:
        dict: yaml content
    """
    # load yml config file
    import yaml
    import jinja2
    with open(path, 'r') as file:
        content = yaml.safe_load(file)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
    while re.search(r'\{\{.*\}\}', str(content)):    # in case there are nested jinja templates
        template = env.from_string(str(content))
        content = yaml.safe_load(template.render(**content))
    return content


def get_main_caller() -> str:
    """Get the main script name that calls the function/script."""
    for frame in inspect.stack():
        module = inspect.getmodule(frame[0])
        if module is not None and module.__name__ == '__main__':
            # for terminal mode
            return Path(module.__file__).stem
        else:
            # for VS Code interactivate window mode
            if frame[0].f_locals.get('__vsc_ipynb_file__'):
                return Path(frame[0].f_locals['__vsc_ipynb_file__']).stem
    return 'Unknown'


def check_user_input(info: str, valid_input: list, hint: str) -> str:
    """
    Check user input. Will keep asking until a valid input is entered.
    """
    while True:
        _ = input(info).upper()
        if _ in valid_input:
            break
        else:
            print(f'Invalid input, please enter a valid response from {valid_input} ({hint}).')
    return _


def get_run_id(timestamp: str, lob: str, cov: str, suffix: str) -> str:
    """
    Generate run_id based on timestamp, lob, cov, and suffix.


    Args:
        timestamp: timestamp in the format 'YYYYMMDD_HHMMSSSSSSS'
        cov: coverage, e.g. 'PD', 'CC', 'PD_C6'. This is peril for HOME.
        lob: line of business, e.g. 'AUTO', 'HOME'
        suffix: step name or other identifier
    Returns:
        str: run_id in the format 'YYYYMMDD_HHMMSSSSSSS_LOB_COV_SUFFIX'
        Example: '20250113_154106514338_AUTO_PD_c6_step1_4_scoring'
    """
    return f"{timestamp}_{lob}_{cov}_{suffix}"


def decode_run_id(run_id: str, component: str = None) -> str|dict:
    """
    Decode run_id to timestamp, lob, cov, and suffix.
    Example: '20250113_154106514338_AUTO_PD_c6_step1_4_scoring'.


    Args:
        run_id: run_id to decode. Must be output of get_run_id()
        component: component to return. Must be 'timestamp', 'lob', 'cov', 'suffix', or None.


    Returns:
        str|dict: decoded component or all components as a dict


    Raises:
        AssertionError: if component is not one of 'timestamp', 'lob', 'cov', 'suffix', or None
        AssertionError: if run_id does not have all components
    """
    assert component in ['timestamp', 'lob', 'cov', 'suffix', None], \
        f"component must be one of 'timestamp', 'lob', 'cov', 'suffix', or None, got {component}."
    run_id_comp = run_id.split('_')
    assert len(run_id_comp) >= 5, 'run_id does not have all components: {run_id}'


    out = {
        'timestamp': '_'.join(run_id_comp[:2]),
        'lob': run_id_comp[2],
        'cov': run_id_comp[3],
        'suffix': '_'.join(run_id_comp[4:]),
    }


    return out[component] if component else out


def get_latest_run_id(run_folder: str, matchkey: str) -> str:
    """
    Get the latest run_id from the run_folder based on the matchkey.


    Example:
        >>> run_folder = 'project_path/output'
        >>> matchkey = '*_AUTO_CC_c1_1_binning_fit'
        get_latest_run_id(run_folder, matchkey) -> '20250113_154106514338_AUTO_CC_c1_1_binning_fit'


    Args:
        run_folder: path to the run folder
        matchkey: keyword to match the run_id
    Returns:
        latest run_id
    """
    run_id_list = glob.glob(f"{run_folder}/{matchkey}")
    if not run_id_list:
        raise FileNotFoundError(
            f'No file/directory found in {run_folder} with keyword {matchkey}.')
    run_id_list.sort(key=os.path.getmtime, reverse=True)  # grab the latest pipeline
    return Path(run_id_list[0]).stem


def get_datehm_from_run_id(run_id: str) -> str:
    """
    Get date and hour-minute from run_id.


    Example:
        >>> run_id = '20250113_154106514338_AUTO_PD_c6_step1_4_scoring'
        >>> get_datehm_from_run_id(run_id)
        '202501131541'


    Args:
        run_id (str): run_id
    Returns:
        str: date and hour-minute
    """
    s_timestamp = decode_run_id(run_id, 'timestamp')
    s_date, s_hm = s_timestamp.split('_')
    return f'{s_date}{s_hm[:4]}'


def input_w_timeout(prompt_message: str, default_input:str = '', timeout: int = 3600) -> str:
    """
    Prompt the user for input with a timeout.
    Works in interactive terminal, debugger, and non-interactive environments.


    Args:
        prompt_message (str): The message to display to the user.
        default_input (str): The default input to return if the user does not respond.
        timeout (int): The timeout in seconds. defrault is 3600 seconds (1 hour).


    Returns:
        str: The user input or the default input if the user does not respond.
    """
    import threading
    import time
    import tkinter as tk
    import textwrap
    import sys
    user_input = [None]
    timeout_flag = [False]
    non_interactive = [False]


    def _get_input_terminal() -> None:
        """
        Get input from the terminal with a timeout.
        """
        user_input[0] = input(prompt_message)


    def _get_input_popup(prompt_message: str, timeout: int = 10) -> None:
        """
        Get input using a popup window with timeout (for interactive terminal environments).
        Keeps the popup window on top of all other applications.


        Args:
            prompt_message (str): The message to display to the user.
            timeout (int): The timeout in seconds.
        Returns: None
        """
        root = tk.Tk()
        root.withdraw()  # Hide the main Tkinter window


        ## SETUP THE POPUP WINDOW
        # Create a Toplevel window for the popup
        toplevel_window = tk.Toplevel(root)
        toplevel_window.attributes('-topmost', True)  # Keep the toplevel window on top
        toplevel_window.focus_force()  # Force focus on the toplevel window
        toplevel_window.title(f"{timeout} seconds to response:")  # Set the title of the popup window


        # Set the size of the popup window
        window_width = 400
        window_height = 300
        # Calculate the center position
        screen_width = toplevel_window.winfo_screenwidth()
        screen_height = toplevel_window.winfo_screenheight()
        position_x = (screen_width // 2) - (window_width // 2)
        position_y = (screen_height // 2) - (window_height // 2)
        # Set the geometry of the popup window
        toplevel_window.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")


        # Add a label and an entry widget for user input
        # Wrap the prompt_message by splitting at white spaces for better readability
        wrapped = []
        for line in prompt_message.splitlines():
            wrapped.extend(textwrap.wrap(line, width=50))
        wrapped_message = '\n'.join(wrapped)
        label = tk.Label(toplevel_window, text=wrapped_message, font=("Arial", 12))
        label.pack(pady=20)
        entry = tk.Entry(toplevel_window, font=("Arial", 14), width=30)
        entry.pack(pady=20)
        entry.focus_set()  # Set focus to the entry box


        ## SETUP THE EVENT HANDLERS
        def _close_popup() -> None:
            """
            Close the popup after the timeout or when the user submits input.
            """
            timeout_flag[0] = True
            toplevel_window.destroy()


        def _submit_input(event=None) -> None:
            """
            Capture the user's input and close the popup.
            """
            user_input[0] = entry.get()
            toplevel_window.destroy()


        # Create a frame to hold the buttons side by side
        button_frame = tk.Frame(toplevel_window)
        button_frame.pack(pady=10)


        # Add a button to submit the input
        submit_button = tk.Button(button_frame, text="Submit/Return", command=_submit_input, font=("Arial", 12))
        submit_button.pack(side=tk.LEFT, padx=5)
        # Bind the Enter key to submit the input
        entry.bind('<Return>', _submit_input)


        # Add a button to close the popup
        close_button = tk.Button(button_frame, text="Close/Escape", command=_close_popup, font=("Arial", 12))
        close_button.pack(side=tk.LEFT, padx=5)
        # Bind the Escape key to close the popup
        toplevel_window.bind('<Escape>', lambda event: _close_popup())
        # if window is closed, set timeout_flag to True
        toplevel_window.protocol("WM_DELETE_WINDOW", _close_popup)


        # Schedule the popup to close after the timeout
        toplevel_window.after(timeout * 1000, _close_popup)


        toplevel_window.mainloop()  # Start the Tkinter main loop


    from IPython import get_ipython
    if get_ipython():
        # if running in Jupyter or IPython, use popup window
        input_thread = threading.Thread(target=_get_input_popup, args=(prompt_message, timeout))
        input_thread.daemon = True
        input_thread.start()
        # Wait for the thread to complete or timeout
        start_time = time.time()
        while input_thread.is_alive():
            if timeout_flag[0] or user_input[0] is not None:
                break
            time.sleep(0.1)  # Avoid busy-waiting
    elif sys.gettrace(): # sys.stdin.isatty(): # interactive terminal
        # if running in debugger mode, use input
        input_thread = threading.Thread(target=_get_input_terminal)
        input_thread.daemon = True
        input_thread.start()
        # Wait for the thread to complete or timeout
        start_time = time.time()


        while input_thread.is_alive():
            if time.time() - start_time > timeout:
                timeout_flag[0] = True
                break
            time.sleep(0.1)  # Avoid busy-waiting
    else:
        # if running in non-interactive environment (e.g., script or non-interactive terminal), skip input
        non_interactive[0] = True


    # print(user_input, timeout_flag, non_interactive)


    if user_input[0] != None:
        if user_input[0].strip() != '':
            # If the user provided input, return it
            logger.info(f"{prompt_message}\n"
                        f"User input: '{user_input[0].strip()}'")
            return user_input[0].strip()
        else:
            # If the user provided empty input, return the default input
            logger.info(f"{prompt_message}\n"
                        f"User input was empty. Using default input: '{default_input}'.")
            return default_input
    else:
        if non_interactive[0]:
            logger.info(f"{prompt_message}\n"
                        f"Non-interactive environment detected. Skipping input with default: '{default_input}'.")
        if timeout_flag[0]:
            logger.info(f"{prompt_message}\n"
                        f"Timeout reached after {timeout} seconds or user choose not to response. "
                        f"Skipping input with default: '{default_input}'.")


        return default_input


def get_git_commit_hash() -> str:
    """
    Get the current git commit hash.


    Returns:
        git commit hash
    """
    import subprocess
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        branch_name = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('utf-8').strip()
        # logger.info(f'Git branch: {branch_name}, commit hash: {commit_hash}')
    except Exception as e:
        logger.warning(f'Could not get git commit hash: {e}')
        commit_hash = 'unknown'
        branch_name = 'unknown'
    return branch_name, commit_hash


def load_pickle_from_run(run_id: str, pattern: str) -> Any:
    """
    Load pickle from the specified run.


    Args:
        run_id: Run id to load the pickle from. Including path.
        pattern: Pattern to match the pickle file. For example, '*pipeline_bindata*'.
    Returns:
        The loaded object.
    """
    pickle_list = glob.glob(f"{run_id}/*/{pattern}.pkl")
    pickle_list.sort(key=os.path.getmtime, reverse=True)  # grab the latest pipeline
    if len(pickle_list) == 0:
        logger.error(f'No pickle object found for run id {run_id} with pattern {pattern}.')
        raise ValueError(f'No pickle object found for run id {run_id} with pattern {pattern}.')
    elif len(pickle_list) > 1:
        logger.warning(f'Multiple pickle objects found for run id {run_id} with pattern {pattern}. '
                       f'\nUsing the latest one (sorted descending by modification time):'
                       f'\n{[Path(p).name for p in pickle_list]}.')
    pickle_save_path = pickle_list[0]
    result = pd.read_pickle(f"{pickle_save_path}")
    logger.info(f'Loaded pickle from {pickle_save_path}.')
    return result, pickle_save_path


def get_timestamp_from_log(folder: str) -> str:
    """Get the start and end timestamp of a run by parsing its log file.
    Searches for the first and last 100 lines of the log file for timestamps in the format 'YYYY-MM-DD HH:MM:SS'.


    Args:
        folder: path to the run folder containing the log file
    Returns:
        start_time and end_time as datetime objects.
        None if not able to find or parse the timestamps.
        If log file is not found or timestamps cannot be parsed, return None for both.
    """
    from collections import deque
    log_path = Path(folder) / 'log.log'
    if not log_path.exists():
        logger.warning(f"Log file not found in {folder}. Cannot decode timestamp.")
        return None, None
    # find the first and last timestamp in log
    # format [...] YYYY-MM-DD HH:MM:SS ...
    date_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
    log_pattern = rf'\[.*?\]\s({date_pattern})\s.*'
    with open(log_path, 'r') as f:
        # Read first 100 lines (or fewer if file is shorter)
        first_100 = list(islice(f, 100))
        if len(first_100) < 100:
            # If the file has fewer than 100 lines, use the same lines for last_100
            last_100 = first_100
        else:
            last_100  = list(deque(f, maxlen=100))
        start_time, end_time = None, None
        for line in first_100:
            match = re.search(log_pattern, line)
            if match:
                start_time = match.group(1)
                break
        for line in reversed(last_100):
            match = re.search(log_pattern, line)
            if match:
                end_time = match.group(1)
                break
    start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') if start_time else None
    end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') if end_time else None
    return start_time, end_time


def get_value_from_config(folder: str, key: str) -> str:
    """Get a specific config value from the config file in a run folder.


    Args:
        folder: path to the run folder containing the config file
        key: the key of the config value to retrieve
    Returns:
        The value corresponding to the key in the config file,
        or 'UNKNOWN' if the key is not found or the config file is missing.
    """
    config_path = glob.glob(f'{folder}/config_*.json')
    config_path.sort(key=os.path.getmtime, reverse=True)  # in case there are multiple config files, use the latest one
    if len(config_path) == 0:
        logger.warning(f"Config file not found in {folder}.")
        return 'UNKNOWN'
    config_path = config_path[0]
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get(key, 'UNKNOWN')


def get_parent_up_to(full_path: str, target_folder: str) -> str:
    """
    Get the parent directory of the full_path up to the target_folder.
    Path in and out are string.


    Args:
        full_path: the full path to start from
        target_folder: the target folder name to search for in the path
    Returns:
        The parent directory up to the target_folder,
        or an empty string if the target_folder is not found in the path.
    """
    path = Path(full_path).resolve()


    # Check current directory and all parents
    for p in [path] + list(path.parents):
        if p.name == target_folder:
            return str(p)


    return "" # Target folder was not found in the path
