import argparse

from src.regression import reg_pipeline
from src.classification import clf_pipeline


arg_parser = argparse.ArgumentParser(description="ML Assignemnt 1")
arg_parser.add_argument(
    "-r", "--run_regression_tasks", action=argparse.BooleanOptionalAction, default=False
)
arg_parser.add_argument(
    "-c",
    "--run_classification_tasks",
    action=argparse.BooleanOptionalAction,
    default=False,
)
arg_parser.add_argument(
    "-a", "--run_all", action=argparse.BooleanOptionalAction, default=False
)


if __name__ == "__main__":
    args = arg_parser.parse_args()

    if args.run_regression_tasks or args.run_all:
        reg_pipeline()

    if args.run_classification_tasks or args.run_all:
        clf_pipeline()
