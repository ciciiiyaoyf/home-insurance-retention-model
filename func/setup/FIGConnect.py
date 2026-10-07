# -*- coding: utf-8 -*-
"""Lazy public Snowflake connections, preserving the original FIGConnect API.

Credentials come from process environment variables or a standard Snowflake
connections.toml profile. Importing this module or constructing a connection
never checks credentials, opens a browser, or contacts a warehouse.
"""
from __future__ import annotations

import inspect
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from func.utils import check_logger, get_git_commit_hash

logger = logging.getLogger(__name__)

_ENV_PARAMETERS = {
    "account": "SNOWFLAKE_ACCOUNT",
    "user": "SNOWFLAKE_USER",
    "password": "SNOWFLAKE_PASSWORD",
    "authenticator": "SNOWFLAKE_AUTHENTICATOR",
    "warehouse": "SNOWFLAKE_WAREHOUSE",
    "role": "SNOWFLAKE_ROLE",
    "database": "SNOWFLAKE_DATABASE",
    "schema": "SNOWFLAKE_SCHEMA",
    "connection_name": "SNOWFLAKE_CONNECTION_NAME",
    "private_key_file": "SNOWFLAKE_PRIVATE_KEY_FILE",
    "private_key_file_pwd": "SNOWFLAKE_PRIVATE_KEY_FILE_PWD",
    "token": "SNOWFLAKE_TOKEN",
}


def snowflake_connection_parameters(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read and validate configuration only; do not import a driver or connect.

    Explicit nonempty parameters override environment values. A named profile
    supplies authentication through the connector's standard connections.toml.
    The connector remains responsible for validating account authentication.
    """
    values: dict[str, Any] = {
        key: os.environ[name] for key, name in _ENV_PARAMETERS.items()
        if os.environ.get(name, "").strip()
    }
    values.update({key: value for key, value in (overrides or {}).items()
                   if value is not None and value != ""})
    values.setdefault("database", "INSURANCE_ANALYTICS")
    values.setdefault("schema", "RETENTION")
    for key in ("database", "schema"):
        value = str(values[key]).strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_$]*", value):
            raise ValueError(f"SNOWFLAKE_{key.upper()} must be an unquoted SQL identifier.")
        values[key] = value
    if "login_timeout" not in values:
        try:
            values["login_timeout"] = int(os.getenv("SNOWFLAKE_LOGIN_TIMEOUT", "60"))
        except ValueError as exc:
            raise ValueError("SNOWFLAKE_LOGIN_TIMEOUT must be a positive integer.") from exc
    if isinstance(values["login_timeout"], bool) or int(values["login_timeout"]) < 1:
        raise ValueError("SNOWFLAKE_LOGIN_TIMEOUT must be a positive integer.")
    if values.get("private_key_file"):
        values["private_key_file"] = str(Path(values["private_key_file"]).expanduser())
    if not values.get("connection_name"):
        missing = [name for name in ("account", "user") if not values.get(name)]
        if missing:
            raise ValueError("Set " + ", ".join(_ENV_PARAMETERS[name] for name in missing)
                             + " or SNOWFLAKE_CONNECTION_NAME before querying Snowflake.")
        authenticator = str(values.setdefault("authenticator", "snowflake")).lower()
        if authenticator in {"snowflake", "username_password_mfa"} and not (
            values.get("password") or values.get("private_key_file") or values.get("private_key")
        ):
            raise ValueError("Set SNOWFLAKE_PASSWORD, a private key, or an explicit authenticator.")
        if authenticator == "oauth" and not values.get("token"):
            raise ValueError("SNOWFLAKE_AUTHENTICATOR=oauth requires SNOWFLAKE_TOKEN.")
        if authenticator == "snowflake_jwt" and not (
            values.get("private_key_file") or values.get("private_key")
        ):
            raise ValueError("SNOWFLAKE_AUTHENTICATOR=SNOWFLAKE_JWT requires a private key.")
    return values


class FIGConnect:
    """Snowflake/Snowpark wrapper with lazy credentials and query provenance.

    The historical class/file name is retained for existing project imports.
    SQLAlchemy ``engine``/``cnxn`` and native ``snow_conn`` become available
    after an explicit ``create_conn`` or first query, and share one session.
    """
    __allowed = {
        "role": str, "warehouse": str, "database": str, "schema": str,
        "mode": str, "username": str, "password": str, "timeout": int,
        "memory_reduction": bool, "account": str, "authenticator": str,
        "connection_name": str, "private_key_file": str,
        "private_key_file_pwd": str, "token": str, "connection_parameters": dict,
    }

    def __init__(self, server="SNOWFLAKE", **kwargs):
        check_logger(logger)
        self.server = str(server).upper()
        if self.server not in {"SNOWFLAKE", "SNOWPARK"}:
            raise ValueError("Supported public servers are SNOWFLAKE and SNOWPARK.")
        for name in ("database", "schema", "role", "warehouse", "username", "password",
                     "account", "authenticator", "connection_name", "private_key_file",
                     "private_key_file_pwd", "token"):
            setattr(self, name, "")
        self.mode = "read"
        self.timeout = None
        self.memory_reduction = False
        self.connection_parameters = {}
        self.engine = self.cnxn = self.snow_conn = self.session = None
        self.query = ""
        self.data = None
        self.queries = {}
        self.time = None
        self.query_name = ""
        self.query_id = ""
        self.original_dtypes = ()
        for key, value in kwargs.items():
            if key not in self.__allowed or not isinstance(value, self.__allowed[key]):
                raise TypeError(f"Invalid connection parameter or type: {key}")
            setattr(self, key, value)
        if self.mode not in {"read", "write"}:
            raise ValueError("mode must be read or write.")

    def show_options(self):
        return {"servers": ["SNOWFLAKE", "SNOWPARK"],
                "configuration": "Environment variables or Snowflake connections.toml"}

    def _connection_parameters(self, large_query=False):
        values = dict(self.connection_parameters)
        for key in _ENV_PARAMETERS:
            attribute = "username" if key == "user" else key
            value = getattr(self, attribute, "")
            if value:
                values[key] = value
        if self.timeout is not None:
            values["login_timeout"] = self.timeout
        values.setdefault("client_fetch_use_mp", bool(large_query))
        values.setdefault("client_prefetch_threads", 10 if large_query else 4)
        return snowflake_connection_parameters(values)

    def _snowflake_cnxn(self, verbose=20, large_query=False):
        parameters = self._connection_parameters(large_query)
        import snowflake.connector
        from sqlalchemy import create_engine
        # creator keeps passwords/keys/tokens out of SQLAlchemy URL strings and
        # supports the connector's standard connection_name authentication.
        self.engine = create_engine(
            "snowflake://", creator=lambda: snowflake.connector.connect(**parameters),
            pool_pre_ping=True,
        )
        try:
            self.cnxn = self.engine.connect()
            self.snow_conn = self.cnxn.connection.driver_connection
        except Exception:
            self.close_conn(verbose=0)
            raise
        if verbose >= 20:
            logger.info("Snowflake session opened.")

    def _snowpark_cnxn(self, verbose=20):
        parameters = self._connection_parameters()
        try:
            from snowflake.snowpark import Session
        except ImportError as exc:
            raise RuntimeError("Install snowflake-snowpark-python for explicit Snowpark use.") from exc
        self.session = Session.builder.configs(parameters).create()
        if verbose >= 20:
            logger.info("Snowpark session opened.")

    def create_conn(self, verbose=20, large_query=False):
        if self.server == "SNOWFLAKE" and self.cnxn is None:
            self._snowflake_cnxn(verbose, large_query)
        elif self.server == "SNOWPARK" and self.session is None:
            self._snowpark_cnxn(verbose)
        return self

    def close_conn(self, verbose=20):
        for attribute, method in (("session", "close"), ("cnxn", "close"), ("engine", "dispose")):
            resource = getattr(self, attribute, None)
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception:
                    logger.warning("Could not close %s cleanly.", attribute)
                finally:
                    setattr(self, attribute, None)
        self.snow_conn = None
        if verbose >= 20:
            logger.info("Connections closed.")

    def df_memory_reduce(self, df):
        start_mem_usg = df.memory_usage().sum() / 1024**2
        # print("Memory usage of properties dataframe is :",start_mem_usg," MB")
        cols = list(df.select_dtypes(include=['number']).columns)
        for col in cols:
            # # Print current columinvalue type
            # print("******************************")
            # print("Column: ",col)
            # print("dtype before: ",df[col].dtype)


            # make variables for Int, max and min
            IsInt = True
            maxvalue = df[col].max()
            minvalue = df[col].min()
            if pd.isnull(minvalue):
                minvalue = 0
            if pd.isnull(maxvalue):
                maxvalue = 0


            # test if columinvalue can be converted to an integer
            try:
                asint = df[col].fillna(0).astype(pd.Int64Dtype())
            except TypeError:  # Int64Dtype does not accept inaccurate casts
                IsInt = False
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if minvalue >= 0:
                    if maxvalue < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(pd.UInt8Dtype())
                    elif maxvalue < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(pd.UInt16Dtype())
                    elif maxvalue < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(pd.UInt32Dtype())
                    else:  # sqlalchemy doesn't support UInt64
                        df[col] = df[col].astype(pd.Int64Dtype())
                else:
                    if minvalue > np.iinfo(np.int8).min and maxvalue < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(pd.Int8Dtype())
                    elif minvalue > np.iinfo(np.int16).min and maxvalue < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(pd.Int16Dtype())
                    elif minvalue > np.iinfo(np.int32).min and maxvalue < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(pd.Int32Dtype())
                    elif minvalue > np.iinfo(np.int64).min and maxvalue < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(pd.Int64Dtype())


            # Make float datatypes 64 bit
            elif maxvalue < 2**20:
                # if max value is greater than 2^24, then float32 is not precise enough: https://en.wikipedia.org/wiki/Single-precision_floating-point_format
                # considering the sum of the column, use 2**20 as the threshold
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)


            # # Print new columinvalue type
            # print("dtype after: ",df[col].dtype)
            # print("******************************")


        # # Print final result
        # print("___MEMORY USAGE AFTER COMPLETION:___")
        # mem_usg = df.memory_usage().sum() / 1024**2
        # print("Memory usage is: ",mem_usg," MB")
        # print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
        return df


    def process_data(self, df, lowercase_cols=False, case_sensitive=False, allow_duplicates=False):
        # check for duplicate column names
        import warnings
        if df.columns.duplicated().sum() > 0:
            dupe_names = list(df.columns[df.columns.duplicated()])
            logger.warn(f"Dataframe has duplicate column names.\n{dupe_names}")
            # warning.warn(f"Dataframe has duplicate column names.\n{dupe_names}")
            if not allow_duplicates and len(dupe_names) > 0:
                logger.info(f'Dropping duplicate columns: \n{dupe_names}')
                df = df.loc[:,~df.columns.duplicated()]
                dupe_names = list(df.columns[df.columns.duplicated()])
                if df.columns.duplicated().sum() > 0:
                    raise Exception("Failed to drop duplicate columns:", dupe_names)
            else:
                logger.warn(f"Dataframe has duplicate column names. {dupe_names}")


        # convert column names to upper/lower case
        if not case_sensitive:
            # SQLAlchemy considers lowercase columns case-insensitive
            # Transform to upper to mainting Snowflake convention
            df.columns = df.columns.str.upper()
        if lowercase_cols:
            df.columns = df.columns.str.lower()
        # reduce memory usage
        if self.memory_reduction:
            self.original_dtypes = df.dtypes
            df = self.df_memory_reduce(df)
        return df




    def query_data(self, query, query_name=None, chunksize=None, lowercase_cols=False, verbose=20,
                   case_sensitive=False, allow_duplicates=False, cat_cols=None):
        """Execute one query, retaining its real query ID and streaming rows if requested."""
        if chunksize is not None and (isinstance(chunksize, bool) or int(chunksize) < 1):
            raise ValueError("chunksize must be a positive integer.")
        self.original_dtypes = ()
        self.create_conn(verbose=verbose, large_query=chunksize is not None)
        self.query = query
        self.query_name = query_name or ""
        started = time.time()
        if self.server == "SNOWPARK":
            # Snowpark executes through its underlying native connection too;
            # using it here preserves exactly the same result/provenance API.
            native_connection = self.session.connection
        else:
            native_connection = self.snow_conn
        cursor = native_connection.cursor()
        try:
            cursor.execute(query)
            self.query_id = cursor.sfqid
        except Exception:
            cursor.close()
            raise
        columns = [item[0] for item in (cursor.description or [])]

        def process(frame):
            result = self.process_data(frame, lowercase_cols, case_sensitive, allow_duplicates)
            for name in cat_cols or ():
                column = name if case_sensitive else str(name).upper()
                if lowercase_cols:
                    column = str(column).lower()
                if column in result:
                    result[column] = result[column].astype("category")
            return result

        if chunksize is not None:
            def chunks():
                try:
                    while True:
                        rows = cursor.fetchmany(int(chunksize))
                        if not rows:
                            break
                        yield process(pd.DataFrame.from_records(rows, columns=columns))
                finally:
                    cursor.close()
            self.data = chunks()
            # Record the receipt immediately, as callers inspect query_id
            # before consuming the iterator. The iterator owns cursor cleanup.
            self.time = time.time() - started
            self.save_info()
            return self.data
        try:
            from snowflake.connector.errors import NotSupportedError
            try:
                arrow = cursor.fetch_arrow_all(force_return_table=True)
                frame = arrow.to_pandas() if arrow is not None else pd.DataFrame(columns=columns)
            except NotSupportedError:
                frame = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
            self.data = process(frame)
            self.time = time.time() - started
            self.save_info()
            return self.data
        finally:
            cursor.close()

    def test_connection(self):
        """Explicit connectivity test; authentication/query errors propagate."""
        return self.query_data("SELECT 1 AS CONNECTION_OK")

    def write_data(self, df: pd.DataFrame, table_name, verbose=20,
                   quote_identifiers=False, comment="", **kwargs):
        """Explicitly write a DataFrame using Snowflake's supported bulk writer."""
        if self.server != "SNOWFLAKE":
            raise ValueError("write_data requires a SNOWFLAKE connection.")
        self.create_conn(verbose=verbose)
        from snowflake.connector.pandas_tools import write_pandas
        self.data = self.df_memory_reduce(df.copy()) if self.memory_reduction else df
        self.query_name = f"write_{table_name}"
        self.query = f"write_pandas: {table_name}"
        self.query_id = ""
        started = time.time()
        options = {"compression": "snappy", "chunk_size": 50000, "use_logical_type": True, **kwargs}
        result = write_pandas(self.snow_conn, self.data, table_name,
                              quote_identifiers=quote_identifiers, **options)
        if not result[0]:
            raise RuntimeError("Snowflake write_pandas reported an unsuccessful upload.")
        if comment:
            identifier = str(table_name)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*){0,2}", identifier):
                raise ValueError("Table comments require a safe unquoted table identifier.")
            cursor = self.snow_conn.cursor()
            try:
                cursor.execute("COMMENT ON TABLE IDENTIFIER(%s) IS %s", (identifier, comment))
                self.query_id = cursor.sfqid
            finally:
                cursor.close()
        self.time = time.time() - started
        self.save_info()
        return result

    def save_info(self, slow_metadata=False):
        # [print(x.filename, x.function) for x in inspect.stack()]
        module_path = inspect.stack()[2].filename
        line = inspect.stack()[2].lineno
        git_hash = get_git_commit_hash()[1]
        calling_module = os.path.splitext(os.path.basename(module_path))[0]
        calling_function = inspect.stack()[2].function
        query_path = '.'.join(
            filter(None, [calling_module, calling_function, self.query_name]))
        i = 1
        _query_path = query_path
        while query_path in self.queries:
            query_path = _query_path + f'__{i}'
            i += 1
        self.queries[query_path] = {}
        self.queries[query_path]["sql"] = self.query
        self.queries[query_path]["time"] = self.time
        if slow_metadata:
            self.queries[query_path]["count"] = self.data.count() if isinstance(self.data, pd.DataFrame) else 'Generator'
            self.queries[query_path]["unique"] = self.data.nunique() if isinstance(self.data, pd.DataFrame) else 'Generator'
        self.queries[query_path]["shape"] = self.data.shape if isinstance(self.data, pd.DataFrame) else 'Generator'
        self.queries[query_path]["dtypes"] = self.data.dtypes if isinstance(self.data, pd.DataFrame) else 'Generator'
        self.queries[query_path]["original_dtypes"] = self.original_dtypes
        self.queries[query_path]["mode"] = self.mode
        self.queries[query_path]["path"] = f'{module_path}\nLine: {line}\nGit hash: {git_hash}'
        self.queries[query_path]["id"] = self.query_id



SnowflakeConnect = FIGConnect
