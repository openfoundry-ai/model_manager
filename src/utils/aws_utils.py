def construct_s3_uri(bucket, prefix):
    return f"s3://{bucket}/{prefix}"
