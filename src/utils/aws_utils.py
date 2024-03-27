def construct_s3_uri(bucket: str, prefix: str) -> str:
    return f"s3://{bucket}/{prefix}"


def is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")
