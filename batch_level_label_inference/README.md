# batch_level_label_inference

## 0. Quick Start
* The following contents of this README file shows some of the basic utils of the code in this folder.
* A quick start of running all the codes in `./batch_level_label_inference/` folder is to use `bash ./run.sh` under `./batch_level_label_inference/` directory in terminal. The results of the experiments will be in folder `./exp_result/`.

## 1. Settings:

| Dataset  | Model                |
| -------- | -------------------- |
| mnist    | mlp2                 |
| nuswide  | mlp2                 |
| cifar10  | resnet18 or resnet20 |
| cifar100 | resnet18 or resnet20 |

## 2. Prepare the training data

 Default data directory should be in folder: `./data/` or `./dataset/`

| DATASET  | SOURCE                                                       |
| -------- | ------------------------------------------------------------ |
| MNIST    | provided by pytorch                                          |
| CIFAR10 and CIFAR100 | download the [CIFAR10 and CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and place it in directory: `./dataset/cifar-10-batches-py/` and `./dataset/cifar-100-python/`|
| NUSWIDE  | download the [NUSWIDE](https://pan.baidu.com/s/1FPTlZcXwY2tEqOJViZNU7w)  dataset and place it in directory: `./data/NUSWIDE/` |

## 3. CoAE

* run `./train_CoAE/train_autoencoder.py` to train the CAE encoder and decoder.
* trained CAE models are in folder `./trained_models/`.
* example models are ready in folder `./trained_models/`. ( `batch_level_label_inference, replacement_backdoor_binary, replacement_backdoor_multi` share the same trained CAE encoder and decoder)


## 4. Defense

### 4.1 experiments of label inference attack

| filename | description |
| :- | :- |
| vfl_dlg.py                  | the class of label inference attack algorithm                |
| vfl_dlg_ae.py               | label inference attack under CAE defense                    |
| vfl_dlg_gaussian.py         | label inference attack under dp-gaussian defense             |
| vfl_dlg_grad_spars.py       | label inference attack under gradient sparsification defense |
| vfl_dlg_laplace.py          | label inference attack under dp-laplace defense              |
| vfl_dlg_model_completion.py | label inference attack under ppdl, gc, discrete-sgd, ng mentioned by [Fu](https://github.com/FuChong-cyber/label-inference-attacks)|
| vfl_dlg_no_defense.py       | label inference attack w/o defense                           |

### 4.2 experiments of main attack

| filename | description |
| :- | :- |
| vfl_main_task.py      | the class of VFL main task                       |
| vfl_dlg_ae.py         | main task under CAE defense                     |
| vfl_dlg_gaussian.py   | main  task under dp-gaussian defense             |
| vfl_dlg_grad_spars.py | main  task under gradient sparsification defense |
| vfl_dlg_laplace.py    | main  task under dp-laplace defense              |
| vfl_dlg_model_completion.py |  main  task under ppdl, gc, discrete-sgd, ng mentioned by [Fu](https://github.com/FuChong-cyber/label-inference-attacks)|
| vfl_dlg_no_defense.py | main  task  w/o defense                          |
