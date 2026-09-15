"""Tests using sample data."""

import pytest

from analyzer import Analyzer

CSV_PATH = "user_engagement.csv"


@pytest.fixture(scope="session")
def records(spark):
    """Load the sample data into a DataFrame."""
    return Analyzer(spark).load(CSV_PATH)


def test_load_reads_all_rows(records):
    """CSV shape test."""
    assert records.count() == 6
    assert records.columns == [
        "user_id",
        "timestamp",
        "page",
        "duration_seconds",
    ]


def test_average_duration_per_page(records):
    """Average duration per page test."""
    result = Analyzer.average_duration_per_page(records)

    actual = {row["page"]: row["avg_duration_sec"] for row in result.collect()}
    assert actual == {"home": 25.0, "dashboard": 42.5, "profile": 45.0}


def test_most_engaging_page(records):
    """Most engaging page test."""
    averages = Analyzer.average_duration_per_page(records)

    assert Analyzer.most_engaging_page(averages) == ("profile", 45.0)
