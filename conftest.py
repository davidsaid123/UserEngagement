"""Pytest configuration and shared fixtures."""

import os
import sys

import pytest
from pyspark.sql import SparkSession

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark():
    """Start Spark session for the tests."""
    session = SparkSession.builder.master("local[1]").getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
