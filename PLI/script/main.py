import argparse
import os
import random
import string
from datetime import datetime

from pli.attack.confidence import get_pi, get_pj
from pli.config.config import config_base, config_dataset, config_fedkd
from pli.pipeline.fedkd.pipeline import attack_fedkd


def randomname(n):
    randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    return "".join(randlst)


def add_args(parser):
    parser.add_argument(
        "-t",
        "--fedkd_type",
        type=str,
        default="FedMD",
        help="type of FedKD; FedMD, FedGEMS, or FedGEMS",
    )

    parser.add_argument(
        "--inv_learning_rate", type=float, default=0.00003, help="learning rate"
    )

    parser.add_argument(
        "--num_communication", type=int, default=5, help="number of communication"
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
        "--tot_class_num", type=int, default=1000, help="number of total classes"
    )
    parser.add_argument(
        "--tar_class_num", type=int, default=200, help="number of target classes"
    )

    parser.add_argument(
        "-u",
        "--blur_strength",
        type=int,
        default=10,
        help="strength of blur",
    )

    parser.add_argument(
        "-a",
        "--attack_type",
        type=str,
        default="pli",
        help="type of attack; pli or tbi",
    )

    parser.add_argument("--alpha", type=float, default=5.0, help="alpha")
    parser.add_argument("--gamma", type=float, default=0.03, help="gamma")

    parser.add_argument(
        "--invloss", type=str, default="mse", help="loss function for inversion"
    )

    parser.add_argument(
        "-s",
        "--softmax_tempreature",
        type=float,
        default=3.0,
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
        default=None,
        help="path to the trained model folder",
    )

    parser.add_argument(
        "-b",
        "--ablation_study",
        type=int,
        default=2,
        help="type of ablation study; 0: only local logits with prior-based inference adjusting, \
                                      1: only local logits witout inference adjusting,\
                                      2: paird logits with prior-based inference adjusting",
    )

    parser.add_argument("--use_multi_models", type=int, default=1)

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
    args["num_communication"] = parsed_args.num_communication
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

    if args["fedkd_type"] == "tbi":
        parsed_args.use_multi_models = 0

    args["random_seed"] = parsed_args.random_seed
    args["gamma"] = parsed_args.gamma
    args["only_sensitive"] = parsed_args.data_for_inversion
    args["use_multi_models"] = parsed_args.use_multi_models
    with open(os.path.join(run_dir, "args.txt"), "w") as convert_file:
        convert_file.write(str(args))
    args.pop("random_seed")
    args.pop("gamma")
    args.pop("only_sensitive")
    args.pop("use_multi_models")

    print("Start experiment ...")
    print("dataset is ", args["dataset"])
    print("#classes is ", args["num_classes"])
    print("#target classes is ", args["config_dataset"]["target_celeblities_num"])
    print("pi is ", get_pi(args["num_classes"], args["alpha"]))
    print("pj is ", get_pj(args["num_classes"], args["alpha"]))

    if parsed_args.path_to_model is None:
        if args["dataset"] == "LAG":
            parsed_args.path_to_model = "/content/RPDPKDFL/model/LAG/"
        elif args["dataset"] == "LFW":
            parsed_args.path_to_model = "/content/RPDPKDFL/model/LFW/"
        elif args["dataset"] == "FaceScrub":
            parsed_args.path_to_model = "/content/RPDPKDFL/model/FaceScrub"

    result = attack_fedkd(
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
