# replacement_backdoor_binary

## 0. Quick Start
* The following contents of this README file shows some of the basic utils of the code in this folder.
* A quick start of running all the codes in `./replacement_backdoor_binary/` folder is to use `bash ./run.sh` under `./replacement_backdoor_binary/` directory in terminal. The results of the experiments will be in folders `./experiment_result_*/`.

## 1. Settings:

| Dataset  | Model    |
| -------- | -------- |
| mnist    | mlp2     |
| nuswide  | mlp2     |
| cifar10  | resnet18 |
| cifar20  | resnet18 |

## 2. Prepare the training data

 Default data directory should be: `./dataset/`

| DATASET  | SOURCE                                                       |
| -------- | ------------------------------------------------------------ |
| MNIST    | provided by pytorch                                          |
| CIFAR10 and CIFAR100 | download the [CIFAR10 and CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and place it in directory: `./dataset/cifar-10-batches-py/` and `./dataset/cifar-100-python/`|
| NUSWIDE  | download the [NUSWIDE](https://pan.baidu.com/s/1FPTlZcXwY2tEqOJViZNU7w)  dataset and place it in directory: `./data/NUSWIDE/` |

## 3. CoAE

* run `./train_CoAE/train_autoencoder.py` to train the CAE encoder and decoder.
* trained CAE models are in folder `./trained_models/`.
* example models are in folder `../batch_level_label_inference/trained_models/`. ( `batch_level_label_inference, replacement_backdoor_binary, replacement_backdoor_multi` share the same trained CAE encoder and decoder)


## 4. Attack and Defense

| filename | description |
| :- | :- |
| main.py | main task and backdoor attack accuracy w/o defense and under baseline defenses |
| main_cae.py | main task and backdoor attack accuracy under CAE and DCAE defenses |
| log_parse.py | plottings |
| utils.py | several useful functions |
