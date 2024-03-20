import argparse
import logging
import os
from pathlib import Path

if __name__ == '__main__':
    # Run setup if these files/directories don't already exist
    if (not os.path.exists(os.path.expanduser('~/.aws')) or not os.path.exists('.env')):
        os.system("bash setup.sh")

    parser = argparse.ArgumentParser(
        description="Create, deploy, query against models.",
        epilog="As an alternative to the commandline, params can be placed in a file, one per line, and specified on the commandline like '%(prog)s @params.conf'.",
        fromfile_prefix_chars='@')
    parser.add_argument(
        "--hf",
        help="Deploy a Hugging Face Model.",
        type=str
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    from src.main import main
    main(args, loglevel)
