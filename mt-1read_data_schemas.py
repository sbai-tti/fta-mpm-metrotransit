from pyspark.sql import SparkSession
import os

# Initialize Spark session
spark = SparkSession.builder.appName("ReadParquetSchemas").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Define base data directory
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
raw_data_dir = "data/raw"
if not os.path.exists(raw_data_dir):
    os.makedirs(raw_data_dir)
    
    
def show_parquet_schema(path, label, output_file=None):
    print(f"\nSchema for {label} ({path}):")
    df = spark.read.parquet(path)
    schema_str = df._jdf.schema().treeString()
    print(schema_str)
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"Schema for {label} ({path}):\n")
            f.write(schema_str)
            f.write("\n\n")

# Output schema file path
schema_output_path = os.path.join(data_dir, "raw_data_schemas.txt")
# Clear previous content if exists
open(schema_output_path, "w").close()

# Single parquet files in base directory
# 1. train_cars
train_cars_path = os.path.join(raw_data_dir, "train_cars.parquet")
if os.path.exists(train_cars_path):
    show_parquet_schema(train_cars_path, "train_cars", schema_output_path)
else:
    print(f"{train_cars_path} does not exist.")

# 2. vehicle_train_cars
vehicle_train_cars_path = os.path.join(raw_data_dir, "vehicle_train_cars.parquet")
if os.path.exists(vehicle_train_cars_path):
    show_parquet_schema(vehicle_train_cars_path, "vehicle_train_cars", schema_output_path)
else:
    print(f"{vehicle_train_cars_path} does not exist.")
    
# 3. vehicles
vehicles_path = os.path.join(raw_data_dir, "vehicles.parquet")
if os.path.exists(vehicles_path):
    show_parquet_schema(vehicles_path, "vehicles", schema_output_path)
else:
    print(f"{vehicles_path} does not exist.")
    
# 4. demand_response_summary
demand_response_summary_path = os.path.join(raw_data_dir, "demand_response_summary.parquet")
if os.path.exists(demand_response_summary_path):
    show_parquet_schema(demand_response_summary_path, "demand_response_summary", schema_output_path)
else:
    print(f"{demand_response_summary_path} does not exist.")

# Multiple parquet files in subfolders in the base directory
# 5. passups (recursively find all .parquet files)
passups_path = os.path.join(raw_data_dir, "passups/*/*/*.parquet")
try:
    show_parquet_schema(passups_path, "passups", schema_output_path)
except Exception as e:
    print(f"No passups parquet files found or error: {e}")

# 6. stop_crossings
stop_crossings_path = os.path.join(raw_data_dir, "stop_crossings/*/*/*.parquet")
try:
    show_parquet_schema(stop_crossings_path, "stop_crossings", schema_output_path)
except Exception as e:
    print(f"No stop_crossings parquet files found or error: {e}")

# 7. vehicle_locations
vehicle_locations_path = os.path.join(raw_data_dir, "vehicle_locations/*/*.parquet")
try:
    show_parquet_schema(vehicle_locations_path, "vehicle_locations", schema_output_path)
except Exception as e:
    print(f"No vehicle_locations parquet files found or error: {e}")

# 8. stop_visits 
stop_visits_path = os.path.join(raw_data_dir, "stop_visits/*/*/*.parquet")
try:
    show_parquet_schema(stop_visits_path, "stop_visits", schema_output_path)
except Exception as e:
    print(f"No stop_visits parquet files found or error: {e}")
    
# 9. fare_transactions 
fare_transactions_path = os.path.join(raw_data_dir, "fare_transactions/*/*/*.parquet")
try:
    show_parquet_schema(fare_transactions_path, "fare_transactions", schema_output_path)
except Exception as e:
    print(f"No fare_transactions parquet files found or error: {e}")


# Stop Spark session
spark.stop()
