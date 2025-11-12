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
vehicle_locations_dir = os.path.join(base_dir, "vehicle_locations/tm")

# Define output paths for the three datasets
output_dir = "data/examples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

EXAMPLE_DATE = "2024-07-01"  ## vehicle_locations date only available from 20240701 to 20240711. request 2025 data later

vehloc_path = os.path.join(vehicle_locations_dir, f"{EXAMPLE_DATE.replace('-', '')}.parquet")

def read_parquet_to_df(path, engine="spark", limit=None):
    """
    Read parquet files (supports Spark, pandas, polars).
    - path: glob or directory accepted by Spark (e.g. "data/raw/.../*.parquet")
    - engine: "spark" | "pandas" | "polars"
    - limit: optional int to limit rows (applies at Spark level)
    Returns: Spark DataFrame (if engine=="spark"), pandas.DataFrame, or polars.DataFrame.
    """
    engine = engine.lower()
    # not tested, don't use yet, SBAI, 11/12/2025
    if engine == "spark":
        sdf = spark.read.parquet(path)
        return sdf.limit(limit) if limit is not None else sdf
    elif engine == "pandas":
        sdf = spark.read.parquet(path)
        if limit is not None:
            sdf = sdf.limit(limit)
        return sdf.toPandas()
    # not tested, don't use yet, SBAI, 11/12/2025
    elif engine == "polars": 
        # read via Spark -> pandas -> polars to avoid filesystem differences
        sdf = spark.read.parquet(path)
        if limit is not None:
            sdf = sdf.limit(limit)
        pdf = sdf.toPandas()
        return pl.from_pandas(pdf)
    else:
        raise ValueError(f"unsupported engine: {engine}")

df = read_parquet_to_df(vehloc_path, engine="pandas", limit=None)
df.to_csv(os.path.join(output_dir, f"vehicle_locations_example_{EXAMPLE_DATE}.csv"), index=False)