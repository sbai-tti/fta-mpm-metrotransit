import os
import pandas as pd
import numpy as np
import polars as pl
from pyspark.sql import SparkSession


# Initialize Spark session
spark = SparkSession.builder.appName("ReadParquetSchemas").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Define base data directory
base_dir = "data/raw"
fare_transactions_dir = os.path.join(base_dir, "fare_transactions/cubic")

# Define output paths for the three datasets
output_dir = "data/examples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

EXAMPLE_DATE = "2025-01-01"

transac_path = os.path.join(fare_transactions_dir, f"service_date={EXAMPLE_DATE}", "*.parquet")

def read_parquet_to_df(path, engine="spark", limit=None):
    """
    Read parquet files (supports Spark, pandas, polars).
    - path: glob or directory accepted by Spark (e.g. "data/raw/.../*.parquet")
    - engine: "spark" | "pandas" | "polars"
    - limit: optional int to limit rows (applies at Spark level)
    Returns: Spark DataFrame (if engine=="spark"), pandas.DataFrame, or polars.DataFrame.
    """
    engine = engine.lower()
    if engine == "spark":
        sdf = spark.read.parquet(path)
        return sdf.limit(limit) if limit is not None else sdf
    elif engine == "pandas":
        sdf = spark.read.parquet(path)
        if limit is not None:
            sdf = sdf.limit(limit)
        return sdf.toPandas()
    elif engine == "polars":
        # read via Spark -> pandas -> polars to avoid filesystem differences
        sdf = spark.read.parquet(path)
        if limit is not None:
            sdf = sdf.limit(limit)
        pdf = sdf.toPandas()
        return pl.from_pandas(pdf)
    else:
        raise ValueError(f"unsupported engine: {engine}")

df = read_parquet_to_df(transac_path, engine="pandas", limit=None)
df.to_csv(os.path.join(output_dir, f"fare_transactions_example_{EXAMPLE_DATE}.csv"), index=False)