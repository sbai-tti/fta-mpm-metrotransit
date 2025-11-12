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

# Define output paths for the three datasets
output_dir = "data/examples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

vehicles_path = os.path.join(base_dir, "vehicles.parquet")
vehicle_train_cars_path = os.path.join(base_dir, "vehicle_train_cars.parquet")
train_cars_path = os.path.join(base_dir, "train_cars.parquet")

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

df_vehicles = read_parquet_to_df(vehicles_path, engine="pandas", limit=None)
df_vehicles.to_csv(os.path.join(output_dir, "vehicles_example.csv"), index=False)
df_vehicle_train_cars = read_parquet_to_df(vehicle_train_cars_path, engine="pandas", limit=None)
df_vehicle_train_cars.to_csv(os.path.join(output_dir, "vehicle_train_cars_example.csv"), index=False)
df_train_cars = read_parquet_to_df(train_cars_path, engine="pandas", limit=None)
df_train_cars.to_csv(os.path.join(output_dir, "train_cars_example.csv"), index=False)
