import argparse
import os
import random
import string
from datetime import datetime

from pli.config.config import config_base, config_dataset
from pli.pipeline.fedkd.pipeline_autoencoder import ae_attack_fedkd


def randomname(n):
    randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    return "".join(randlst)


def add_args(parser):
    parser.add_argument(
        "--inv_learning_rate", type=float, default=0.00003, help="learning rate"
    )

    parser.add_argument(
        "--random_seed", type=int, default=42, help="seed of random generator"
    )

    parser.add_argument(
        "-c", "--client_num", type=int, default=10, help="number of clients"
    )

    parser.add_argument(
        "-d", "--dataset", type=str, default="LAG", help="type of dataset; LAG or LFW"
    )

    parser.add_argument(
        "--tot_class_num", type=int, default=300, help="number of total classes"
    )
    parser.add_argument(
        "--tar_class_num", type=int, default=30, help="number of target classes"
    )

    parser.add_argument(
        "-u",
        "--blur_strength",
        type=int,
        default=15,
        help="strength of blur",
    )

    parser.add_argument(
        "--invloss", type=str, default="mse", help="loss function for inversion"
    )

    parser.add_argument(
        "-p",
        "--path_to_datafolder",
        type=str,
        default="/content/lag",
        help="path to the data folder",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="/content/drive/MyDrive/results/",
        help="path to the output folder",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    args = config_base
    args["num_classes"] = parsed_args.tot_class_num
    args["dataset"] = parsed_args.dataset

    args["client_num"] = parsed_args.client_num
    args["inv_lr"] = parsed_args.inv_learning_rate
    args["loss_type"] = parsed_args.invloss

    args["config_dataset"] = config_dataset[args["dataset"]]
    args["config_dataset"]["data_folder"] = parsed_args.path_to_datafolder
    args["config_dataset"].pop("weight_decay")

    if args["dataset"] in ["AT&T", "MNIST", "FaceScrub"]:
        args["config_dataset"]["blur_strength"] = parsed_args.blur_strength

    args["config_dataset"]["target_celeblities_num"] = parsed_args.tar_class_num

    if args["dataset"] in ["MNIST"]:
        args["model_type"] = "LM"
        args["invmodel_type"] = "InvLM"

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id += "_" + randomname(10)
    run_id += f"_{args['dataset']}_autoencoder_{args['client_num']}"
    run_dir = os.path.join(parsed_args.output_folder, run_id)
    os.makedirs(run_dir)

    args["random_seed"] = parsed_args.random_seed
    with open(os.path.join(run_dir, "args.txt"), "w") as convert_file:
        convert_file.write(str(args))
    args.pop("random_seed")

    print("Start experiment ...")
    print("dataset is ", args["dataset"])
    print("#classes is ", args["num_classes"])
    print("#target classes is ", args["config_dataset"]["target_celeblities_num"])

    result = ae_attack_fedkd(
        dataset=args["dataset"],
        client_num=args["client_num"],
        batch_size=args["batch_size"],
        num_classes=args["num_classes"],
        inv_lr=args["inv_lr"],
        num_workers=args["num_workers"],
        loss_type=args["loss_type"],
        config_dataset=args["config_dataset"],
        seed=parsed_args.random_seed,
        output_dir=run_dir,
    )

    print("Results:")
    print(result)

    with open(os.path.join(run_dir, "result.txt"), "w") as convert_file:
        convert_file.write(str(result))
