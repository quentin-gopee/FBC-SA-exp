"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict
import ast
import glob

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(*metrics, directory="", args=None, end_signal=None):
    print("===")
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []
    sorted_accuracies = defaultdict(list)

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        if args.pattern:
            fpath = glob.glob(osp.join(directory, subdir, args.pattern))[-1]
        assert check_isfile(fpath)
        class_distribution = False
        results = False
        per_class_results = False
        output = OrderedDict()
        labeled_dict = None

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if 'Class distribution in the labeled training set:' in line:
                    class_distribution = True
                    continue

                if class_distribution:
                    labeled_dict = ast.literal_eval(line.split('defaultdict(<class \'int\'>, ')[1].rstrip(')'))
                    # argsort the labeled_dict in descending order
                    sorted_labels_idx = np.argsort(list(labeled_dict.values()))[::-1]
                    class_distribution = False
                    continue

                if line == end_signal:
                    results = True
                    continue

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and results:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num
                    continue

                if '=> per-class result' in line:
                    per_class_results = True
                    continue
                
                if per_class_results:
                    match = re.match(r'\* class: (\d+) \(\)\ttotal: \d+\tcorrect: \d+\tacc: ([\d\.]+)%', line)
                    if match:
                        class_id = int(match.group(1))
                        accuracy = float(match.group(2))
                        sorted_accuracies[sorted_labels_idx[class_id]].append(accuracy)
                    continue

        if output:
            outputs.append(output)
        


    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)
    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.1f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* average {key}: {avg:.1f}% +- {std:.1f}%")
        output_results[key] = avg
    print("===")

    print(sorted_accuracies)

    return output_results, sorted_accuracies


def main(args, end_signal):
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    if args.multi_exp:
        final_results = defaultdict(list)
        sorted_accuracies = defaultdict(list)

        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            results, sa = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )

            for key, value in results.items():
                final_results[key].append(value)

            for key, value in sa.items():
                sorted_accuracies[key].extend(value)

        print("Average performance")
        for key, values in final_results.items():
            avg = np.mean(values)
            std = compute_ci95(values) if args.ci95 else np.std(values)
            print(f"* {key}: {avg:.1f}% +- {std:.1f}%")

        print("Imbalanced class performance")
        for key, values in sorted_accuracies.items():
            avg = np.mean(values)
            std = compute_ci95(values) if args.ci95 else np.std(values)
            print(f"* class N{key}: {avg:.1f}% +- {std:.1f}%")

    else:
        parse_function(
            metric, directory=args.directory, args=args, end_signal=end_signal
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to directory")
    parser.add_argument(
        "--ci95",
        action="store_true",
        help=r"compute 95\% confidence interval"
    )
    parser.add_argument(
        "--test-log", action="store_true", help="parse test-only logs"
    )
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--keyword",
        default="accuracy",
        type=str,
        help="which keyword to extract"
    )
    parser.add_argument(
        "--pattern",
        default="",
        type=str,
        help="pattern to match the log file"
    )
    args = parser.parse_args()

    end_signal = "=> result"

    main(args, end_signal)
