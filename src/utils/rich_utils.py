from rich import print


def print_model(text: str):
    print(f"[blue] {text}")


def print_error(text: str):
    print(f"[red] {text}")


def print_success(text: str):
    print(f"[green] {text}")
