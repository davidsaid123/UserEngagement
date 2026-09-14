"""User engagement results."""

from pyspark.sql import DataFrame, SparkSession

from analyzer import Analyzer

INPUT_PATH = "user_engagement.csv"


def format_report(averages: DataFrame) -> str:
    """Build the report table."""
    lines = [
        "Average Duration Per Page:",
        f"| {'page':<12} | {'avg_duration_sec':<18} |",
        "-" * 40,
    ]
    lines += [
        f"| {row['page']:<12} | {row['avg_duration_sec']:<18} |"
        for row in averages.collect()
    ]
    lines += ["-" * 39, "-" * 39]
    return "\n".join(lines)


def main() -> None:
    """Run the report."""
    spark = SparkSession.builder.appName("user_engagement").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    analyzer = Analyzer(spark)
    records = analyzer.clean(analyzer.load(INPUT_PATH))
    averages = analyzer.average_duration_per_page(records).cache()

    print(format_report(averages))

    top = analyzer.most_engaging_page(averages)
    if top:
        print(f"Most engaging page: {top[0]} (average duration: {top[1]} seconds)")

    spark.stop()


if __name__ == "__main__":
    main()