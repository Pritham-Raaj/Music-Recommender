import os
import sys

os.environ.pop('CONTAINER_ID', None)

from dotenv import load_dotenv
load_dotenv()

from pyspark.sql import SparkSession

S3_BUCKET = os.getenv("S3_BUCKET", "musicrec-data")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

S3_INPUT = {
    "tracks": f"s3a://{S3_BUCKET}/raw_tracks.csv",
    "genres": f"s3a://{S3_BUCKET}/raw_genres.csv",
    "artists": f"s3a://{S3_BUCKET}/raw_artists.csv",
}

LOCAL_OUTPUT = {
    "tracks": "data/processed/tracks.parquet",
    "genres": "data/processed/genres.parquet",
    "artists": "data/processed/artists.parquet",
}

TRACKS_SCHEMA = """
    track_id INT,
    album_id INT,
    album_title STRING,
    album_url STRING,
    artist_id INT,
    artist_name STRING,
    artist_url STRING,
    artist_website STRING,
    license_image_file STRING,
    license_image_file_large STRING,
    license_parent_id INT,
    license_title STRING,
    license_url STRING,
    tags STRING,
    track_bit_rate INT,
    track_comments INT,
    track_composer STRING,
    track_copyright_c STRING,
    track_copyright_p STRING,
    track_date_created STRING,
    track_date_recorded STRING,
    track_disc_number INT,
    track_duration STRING,
    track_explicit STRING,
    track_explicit_notes STRING,
    track_favorites INT,
    track_file STRING,
    track_genres STRING,
    track_image_file STRING,
    track_information STRING,
    track_instrumental INT,
    track_interest INT,
    track_language_code STRING,
    track_listens INT,
    track_lyricist STRING,
    track_number INT,
    track_publisher STRING,
    track_title STRING,
    track_url STRING
"""

GENRES_SCHEMA = """
    genre_id INT,
    genre_color STRING,
    genre_handle STRING,
    genre_parent_id INT,
    genre_title STRING
"""

ARTISTS_SCHEMA = """
    artist_id INT,
    artist_active_year_begin INT,
    artist_active_year_end INT,
    artist_associated_labels STRING,
    artist_bio STRING,
    artist_comments INT,
    artist_contact STRING,
    artist_date_created STRING,
    artist_donation_url STRING,
    artist_favorites INT,
    artist_flattr_name STRING,
    artist_handle STRING,
    artist_image_file STRING,
    artist_images STRING,
    artist_latitude DOUBLE,
    artist_location STRING,
    artist_longitude DOUBLE,
    artist_members STRING,
    artist_name STRING,
    artist_paypal_name STRING,
    artist_related_projects STRING,
    artist_url STRING,
    artist_website STRING,
    artist_wikipedia_page STRING,
    tags STRING
"""

def create_spark_session():
    """Create Spark session configured for AWS S3 access."""
    
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not aws_access_key or not aws_secret_key:
        raise ValueError(
            "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY in environment or .env file."
        )
    
    print(f"Initializing Spark with S3 access to region: {AWS_REGION}")
    
    spark = SparkSession.builder \
        .appName("FMA_CSV_to_Parquet") \
        .master("local[*]") \
        .config("spark.jars.packages", 
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.hadoop.fs.s3a.impl", 
                "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key) \
        .config("spark.hadoop.fs.s3a.endpoint", 
                f"s3.{AWS_REGION}.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark

def csv_to_parquet(spark, input_path, output_path, schema_ddl, multiline=False):
    
    print(f"\n{'='*60}")
    print(f"Source (S3):    {input_path}")
    print(f"Output (Local): {output_path}")
    print('='*60)
    
    reader = spark.read \
        .format("csv") \
        .option("header", "true") \
        .schema(schema_ddl)
    
    if multiline:
        reader = reader \
            .option("multiLine", "true") \
            .option("quote", '"') \
            .option("escape", '"') \
            .option("mode", "PERMISSIVE")
    
    df = reader.load(input_path)
    
    count = df.count()
    print(f"Records read: {count:,}")
    
    df.write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(output_path)
    
    print(f"✓ Parquet written to {output_path}")
    
    return {"count": count, "output_path": output_path}

def convert_all(spark):    
    os.makedirs("data/processed", exist_ok=True)
    
    results = {}
    
    results["tracks"] = csv_to_parquet(
        spark,
        S3_INPUT["tracks"],
        LOCAL_OUTPUT["tracks"],
        TRACKS_SCHEMA,
        multiline=True
    )
    
    results["genres"] = csv_to_parquet(
        spark,
        S3_INPUT["genres"],
        LOCAL_OUTPUT["genres"],
        GENRES_SCHEMA,
        multiline=False
    )
    
    results["artists"] = csv_to_parquet(
        spark,
        S3_INPUT["artists"],
        LOCAL_OUTPUT["artists"],
        ARTISTS_SCHEMA,
        multiline=True
    )
    
    return results

def load_parquet(spark):
    
    tracks = spark.read.parquet(LOCAL_OUTPUT["tracks"])
    genres = spark.read.parquet(LOCAL_OUTPUT["genres"])
    artists = spark.read.parquet(LOCAL_OUTPUT["artists"])
    
    print(f"Tracks:  {tracks.count():,} records")
    print(f"Genres:  {genres.count():,} records")
    print(f"Artists: {artists.count():,} records")
    
    return {
        "tracks": tracks,
        "genres": genres,
        "artists": artists
    }