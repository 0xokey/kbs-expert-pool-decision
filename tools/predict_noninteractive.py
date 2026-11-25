import os
import sys
import argparse
import builtins

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_folder", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["test_eval", "predict_new"],
    )
    parser.add_argument("--new_csv", type=str, default=None)
    parser.add_argument("--channel_best_file", type=str, default=None)
    parser.add_argument("--channel_best_epoch", type=str, default=None)
    args = parser.parse_args()

    experiment_folder = os.path.abspath(args.experiment_folder)
    if not os.path.exists(experiment_folder):
        raise FileNotFoundError(f"Missing experiment folder: {experiment_folder}")

    if args.mode == "predict_new" and not args.new_csv:
        raise ValueError("--new_csv is required when --mode predict_new")

    scripted_inputs = ["1", "q"] if args.mode == "test_eval" else ["2", args.new_csv, "q"]
    answers = iter(scripted_inputs)
    original_input = builtins.input

    def fake_input(prompt=""):
        if prompt:
            print(prompt, end="")
        try:
            value = next(answers)
        except StopIteration as exc:
            raise RuntimeError("Ran out of scripted inputs for predict.py") from exc
        print(value)
        return value

    builtins.input = fake_input
    try:
        channel_best_file = predict._resolve_channel_best_file(args)
        predict.predict(experiment_folder, channel_best_file=channel_best_file)
    finally:
        builtins.input = original_input


if __name__ == "__main__":
    main()
