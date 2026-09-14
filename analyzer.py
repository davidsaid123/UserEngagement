"""Analysis of user engagement records."""

from typing import Tuple

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


class Analyzer:
    """Calculate engagement metrics from user activity records."""

    def __init__(self, spark: SparkSession) -> None:
        self._spark = spark

    def load(self, path: str) -> DataFrame:
        """Read all rows from the CSV file."""
        return self._spark.read.csv(path, header=True, schema=SCHEMA)

    @staticmethod
    def average_duration_per_page(records: DataFrame) -> DataFrame:
        """Return the average duration per page sorted by duration."""
        return (
            records.groupBy("page")
            .agg(F.avg("duration_seconds").alias("avg_duration_sec"))
            .orderBy(F.col("avg_duration_sec").desc())
        )

    @staticmethod
    def most_engaging_page(averages: DataFrame) -> Tuple[str, float]:
        """Return the top page and its mean duration."""
        row = averages.first()
        return row["page"], row["avg_duration_sec"]