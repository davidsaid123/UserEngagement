"""Analysis of user engagement records."""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

SCHEMA = StructType([
    StructField("user_id", IntegerType()),
    StructField("timestamp", TimestampType()),
    StructField("page", StringType()),
    StructField("duration_seconds", IntegerType()),
])

UNKNOWN_PAGE = "unknown"


class Analyzer:
    """Calculate engagement metrics from user activity records."""

    def __init__(self, spark: SparkSession) -> None:
        self._spark = spark

    def load(self, path: str) -> DataFrame:
        """Read all rows from the CSV file."""
        if not Path(path).exists():
            raise FileNotFoundError(f"No such file: {path}")
        return self._spark.read.csv(path, header=True, schema=SCHEMA)

    @staticmethod
    def clean(records: DataFrame) -> DataFrame:
        """Remove invalid rows, normalize page names, and label missing pages."""
        cleaned = (
            records.filter(F.col("duration_seconds").isNotNull())
            .filter(F.col("duration_seconds") >= 0)
            .withColumn("page", F.trim(F.lower(F.col("page"))))
        )
        return cleaned.withColumn(
            "page",
            F.when(
                F.col("page").isNull() | (F.col("page") == ""), UNKNOWN_PAGE
            ).otherwise(F.col("page")),
        )

    @staticmethod
    def average_duration_per_page(records: DataFrame) -> DataFrame:
        """Return the average duration per page sorted by duration."""
        return (
            records.groupBy("page")
            .agg(F.avg("duration_seconds").alias("avg_duration_sec"))
            .orderBy(F.col("avg_duration_sec").desc(), F.col("page").asc())
        )

    @staticmethod
    def most_engaging_page(averages: DataFrame) -> tuple[str, float] | None:
        """Return the top real page and its mean duration, or None if empty."""
        row = averages.filter(F.col("page") != UNKNOWN_PAGE).first()
        if row is None:
            return None
        return row["page"], row["avg_duration_sec"]