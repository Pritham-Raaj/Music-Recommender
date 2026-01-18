from src.Ingestion import create_spark_session, convert_all


def run_pipeline():
    print("=" * 60)
    print("Music Data ETL Pipeline")
    print("=" * 60)

    spark = create_spark_session()

    raw_parquet = convert_all(spark)
    
      