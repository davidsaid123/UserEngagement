"""Extended tests for edge cases and design decisions."""

from datetime import datetime

import pytest
from pyspark.sql import Row

from analyzer import SCHEMA, Analyzer


def make_df(spark, rows):
    """Build a DataFrame from rows."""
    return spark.createDataFrame(rows, schema=SCHEMA)


def test_empty_data(spark):
    """Empty data test."""
    empty = make_df(spark, [])
    averages = Analyzer.average_duration_per_page(empty)

    assert Analyzer.most_engaging_page(averages) is None


def test_tie_breaks(spark):
    """Tie-break test."""
    rows = [
        Row(
            user_id=1,
            timestamp=datetime(2022, 1, 1),
            page="home",
            duration_seconds=30,
        ),
        Row(
            user_id=2,
            timestamp=datetime(2022, 1, 1),
            page="profile",
            duration_seconds=30,
        ),
    ]
    df = make_df(spark, rows)
    averages = Analyzer.average_duration_per_page(df)

    first_run = Analyzer.most_engaging_page(averages)
    second_run = Analyzer.most_engaging_page(averages)

    assert first_run == second_run


def test_missing_file(spark):
    """Missing file test."""
    analyzer = Analyzer(spark)

    with pytest.raises(FileNotFoundError):
        analyzer.load("does_not_exist.csv")


def test_missing_column_row_handling(spark, tmp_path):
    """Missing column test."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "user_id,timestamp,page,duration_seconds\n"
        "1,2022-01-01 12:00:00,home\n"
        "2,2022-01-01 12:05:00,profile,45\n"
    )

    analyzer = Analyzer(spark)
    df = analyzer.load(str(bad_csv))

    bad_row = df.filter(df.user_id == 1).first()
    assert bad_row["page"] == "home"
    assert bad_row["duration_seconds"] is None


def test_duplicate_rows_are_both_counted(spark):
    """Duplicate rows test."""
    rows = [
        Row(
            user_id=1,
            timestamp=datetime(2022, 1, 1),
            page="home",
            duration_seconds=30,
        ),
        Row(
            user_id=1,
            timestamp=datetime(2022, 1, 1),
            page="home",
            duration_seconds=30,
        ),
    ]
    df = make_df(spark, rows)
    result = Analyzer.average_duration_per_page(df)

    row = result.first()
    assert row["avg_duration_sec"] == 30.0
    assert df.filter(df.page == "home").count() == 2


def test_negative_duration(spark):
    """Negative duration test."""
    rows = [
        Row(
            user_id=1,
            timestamp=datetime(2022, 1, 1),
            page="home",
            duration_seconds=-5,
        ),
        Row(
            user_id=2,
            timestamp=datetime(2022, 1, 1),
            page="home",
            duration_seconds=30,
        ),
    ]
    df = make_df(spark, rows)
    cleaned = Analyzer.clean(df)
    result = Analyzer.average_duration_per_page(cleaned)

    row = result.first()
    assert row["avg_duration_sec"] == 30.0


def test_page_names_are_normalized(spark):
    """Page name normalization test."""
    rows = [
        Row(
            user_id=1,
            timestamp=datetime(2022, 1, 1),
            page="Home",
            duration_seconds=10,
        ),
        Row(
            user_id=2,
            timestamp=datetime(2022, 1, 1),
            page="home",
            duration_seconds=20,
        ),
        Row(
            user_id=3,
            timestamp=datetime(2022, 1, 1),
            page="home ",
            duration_seconds=30,
        ),
    ]
    df = make_df(spark, rows)
    cleaned = Analyzer.clean(df)
    result = Analyzer.average_duration_per_page(cleaned)

    assert result.count() == 1
    assert result.first()["page"] == "home"


def test_malformed_duration(spark, tmp_path):
    """Malformed duration test."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "user_id,timestamp,page,duration_seconds\n"
        "1,2022-01-01 12:00:00,home,abc\n"
        "2,2022-01-01 12:05:00,home,30\n"
    )

    analyzer = Analyzer(spark)
    df = analyzer.load(str(bad_csv))

    bad_row = df.filter(df.user_id == 1).first()
    assert bad_row["duration_seconds"] is None

    cleaned = Analyzer.clean(df)
    result = Analyzer.average_duration_per_page(cleaned)

    row = result.first()
    assert row["avg_duration_sec"] == 30.0


def test_clean(spark):
    """Clean no-op test."""
    rows = [
        Row(
            user_id=1,
            timestamp=datetime(2022, 1, 1),
            page="home",
            duration_seconds=30,
        ),
        Row(
            user_id=2,
            timestamp=datetime(2022, 1, 1),
            page="profile",
            duration_seconds=45,
        ),
    ]
    df = make_df(spark, rows)
    cleaned = Analyzer.clean(df)

    assert cleaned.count() == 2
    actual = {
        row["page"]: row["duration_seconds"] for row in cleaned.collect()
    }
    assert actual == {"home": 30, "profile": 45}


def test_same_timestamp(spark):
    """Same timestamp test."""
    same_time = datetime(2022, 1, 1, 12, 0, 0)
    rows = [
        Row(user_id=1, timestamp=same_time, page="home", duration_seconds=10),
        Row(
            user_id=1,
            timestamp=same_time,
            page="profile",
            duration_seconds=20,
        ),
    ]
    df = make_df(spark, rows)
    result = Analyzer.average_duration_per_page(df)

    assert result.count() == 2


def test_unknown_page_excluded_from_top(spark):
    """Unknown page exclusion test."""
    rows = [
        Row(
            user_id=1,
            timestamp=datetime(2022, 1, 1),
            page="",
            duration_seconds=1000,
        ),
        Row(
            user_id=2,
            timestamp=datetime(2022, 1, 1),
            page=None,
            duration_seconds=999,
        ),
        Row(
            user_id=3,
            timestamp=datetime(2022, 1, 1),
            page="home",
            duration_seconds=10,
        ),
    ]
    df = make_df(spark, rows)
    cleaned = Analyzer.clean(df)
    result = Analyzer.average_duration_per_page(cleaned)

    pages = {row["page"] for row in result.collect()}
    assert "unknown" in pages

    assert Analyzer.most_engaging_page(result) == ("home", 10.0)
