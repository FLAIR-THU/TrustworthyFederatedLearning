import argparse
import json
import os

from pli.pipeline.fedkd.pipeline_only_evaluation import evaluation_fedkd


def add_args(parser):
    parser.add_argument("--run_dir", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    args_path = os.path.join(parsed_args.run_dir, "args.txt")

    with open(args_path, "r") as f:
        args = f.read()

    args = json.loads(
        args.replace("'", '"')
        .replace("<lambda>", "")
        .replace("<", '"')
        .replace(">", '"')
        .replace("None", '"None"')
        .replace("False", '"False"')
        .replace("True", '"True"')
        .replace("inf", '"inf"')
        .replace('""False""', '"False"')
    )

    random_seed = args["random_seed"]
    args.pop("random_seed")

    result = evaluation_fedkd(
        seed=random_seed,
        output_dir=parsed_args.run_dir,
        **args,
    )

    print("Results:")
    print(result)

    with open(
        os.path.join(parsed_args.run_dir, "re_evaluation_result.txt"), "w"
    ) as convert_file:
        convert_file.write(str(result))
