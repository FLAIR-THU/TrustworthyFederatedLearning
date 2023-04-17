import argparse
import os
import random
import string
from datetime import datetime

from pli.attack.confidence import get_pi, get_pj
from pli.config.config import config_base, config_dataset, config_fedkd
from pli.pipeline.fedkd.pipeline_only_prior import attack_prior


def randomname(n):
    randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    return "".join(randlst)


def add_args(parser):
    parser.add_argument(
        "-t",
        "--fedkd_type",
        type=str,
        default="fedgems",
        help="type of FedKD; FedMD, FedGEMS, or FedGEMS",
    )

    parser.add_argument(
        "--inv_learning_rate", type=float, default=0.00003, help="learning rate"
    )

    parser.add_argument(
        "-c", "--client_num", type=int, default=10, help="number of clients"
    )

    parser.add_argument(
        "--random_seed", type=int, default=42, help="seed of random generator"
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
        "-a",
        "--attack_type",
        type=str,
        default="pli",
        help="type of attack; pli or tbi",
    )

    parser.add_argument("--alpha", type=float, default=3.0, help="alpha")
    parser.add_argument("--gamma", type=float, default=0.1, help="gamma")

    parser.add_argument(
        "--invloss", type=str, default="mse", help="loss function for inversion"
    )

    parser.add_argument(
        "-s",
        "--softmax_tempreature",
        type=float,
        default=1.0,
        help="tempreature $\tau$",
    )

    parser.add_argument(
        "--data_for_inversion",
        type=int,
        default=1,
        help="0: both, 1: only sensitive",
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

    parser.add_argument(
        "-m",
        "--path_to_model",
        type=str,
        default="/content/",
        help="path to the trained model folder",
    )

    parser.add_argument(
        "-b",
        "--ablation_study",
        type=int,
        default=0,
        help="type of ablation study; 0:normal(Q=p'_{c_i, j}+p'_{s, j}+\alpha H(p'_s)), \
                                      1:without entropy (Q=p'_{c_i, j}+p'_{s, j})\
                                      2:without p'_{s, j} (Q=p'_{c_i, j}+\alpha H(p'_s))\
                                      3:without local logit (Q=p'_{s, j}+\alpha H(p'_s))\
                                      4:without sensitive flag",
    )

    parser.add_argument("--use_multi_models", type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    args = config_base
    args["num_classes"] = parsed_args.tot_class_num
    args["dataset"] = parsed_args.dataset
    args["fedkd_type"] = parsed_args.fedkd_type

    args["attack_type"] = parsed_args.attack_type
    args["client_num"] = parsed_args.client_num
    args["inv_lr"] = parsed_args.inv_learning_rate
    args["loss_type"] = parsed_args.invloss
    args["alpha"] = parsed_args.alpha

    args["ablation_study"] = parsed_args.ablation_study
    args["inv_tempreature"] = parsed_args.softmax_tempreature

    args["config_dataset"] = config_dataset[args["dataset"]]
    args["config_dataset"]["data_folder"] = parsed_args.path_to_datafolder
    args["config_fedkd"] = config_fedkd[args["fedkd_type"]]
    args["config_fedkd"]["weight_decay"] = args["config_dataset"]["weight_decay"]
    args["config_dataset"].pop("weight_decay")

    if args["dataset"] in ["AT&T", "MNIST", "FaceScrub"]:
        args["config_dataset"]["blur_strength"] = parsed_args.blur_strength

    args["config_dataset"]["target_celeblities_num"] = parsed_args.tar_class_num

    if args["dataset"] in ["MNIST"]:
        args["model_type"] = "LM"
        args["invmodel_type"] = "InvLM"

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id += "_" + randomname(10)
    run_id += f"_{args['dataset']}_{args['fedkd_type']}_{args['client_num']}"
    run_dir = os.path.join(parsed_args.output_folder, run_id)
    os.makedirs(run_dir)

    only_sensitive = parsed_args.data_for_inversion == 1

    args["random_seed"] = parsed_args.random_seed
    args["gamma"] = parsed_args.gamma
    args["only_sensitive"] = parsed_args.data_for_inversion
    args["use_multi_models"] = parsed_args.use_multi_models
    args["only_prior"] = 1
    with open(os.path.join(run_dir, "args.txt"), "w") as convert_file:
        convert_file.write(str(args))
    args.pop("random_seed")
    args.pop("gamma")
    args.pop("only_sensitive")
    args.pop("use_multi_models")
    args.pop("only_prior")

    print("Start experiment ...")
    print("dataset is ", args["dataset"])
    print("#classes is ", args["num_classes"])
    print("#target classes is ", args["config_dataset"]["target_celeblities_num"])
    print("pi is ", get_pi(args["num_classes"], args["alpha"]))
    print("pj is ", get_pj(args["num_classes"], args["alpha"]))

    result = attack_prior(
        seed=parsed_args.random_seed,
        gamma=parsed_args.gamma,
        output_dir=run_dir,
        temp_dir=run_dir,
        model_path=parsed_args.path_to_model,
        only_sensitive=only_sensitive,
        use_multi_models=(parsed_args.use_multi_models == 1),
        **args,
    )

    print("Results:")
    print(result)

    with open(os.path.join(run_dir, "result.txt"), "w") as convert_file:
        convert_file.write(str(result))
