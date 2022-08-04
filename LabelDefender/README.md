# Label Defender

> This folder is the pytorch implementation for [Defending Batch-Level Label Inference and Replacement Attacks in Vertical Federated Learning](https://www.computer.org/csdl/journal/bd/5555/01/09833321/1F8uKhxrvNe).

## 1. Folders List
| Folder | Setting| Attack | Defense | Paper|
|:-:|:-:|:-:|:-:|:-:|
|batch_level_label_inference| VFL, 2-party | Batch-level Label Inference by gradient inversion | Confusional AutoEncoder([CAE](https://arxiv.org/abs/2112.05409)), Disceret-SGD enchanced CAE([DCAE](https://arxiv.org/abs/2112.05409)), baseline defenses | [link](https://arxiv.org/abs/2112.05409) |
|replacement_backdoor_binary| VFL, 2-party | Replacement Backdoor by replacing labels | Confusional AutoEncoder([CAE](https://arxiv.org/abs/2112.05409)), Disceret-SGD enchanced CAE([DCAE](https://arxiv.org/abs/2112.05409)), baseline defenses | [link](https://arxiv.org/abs/2112.05409) |
|replacement_backdoor_multi| VFL, 4-party | Replacement Backdoor by replacing labels | Confusional AutoEncoder([CAE](https://arxiv.org/abs/2112.05409)), Disceret-SGD enchanced CAE([DCAE](https://arxiv.org/abs/2112.05409)), baseline defenses | [link](https://arxiv.org/abs/2112.05409) |

* baseline defenses includes: Differencial Privacy, Gradient Sparsification, Marvell, Priviacy Preserving Deep Learning, Discrete-SGD.


## 2. Folder Contents

### 2.1. batch_level_label_inference
1. the training process of CoAE (in folder `train_CoAE`)
2. the trained models of CAE (in folder `trained_models`)
3. the main task accuracy vs. label inference task accuracy under various defense strategies
4. More detail information please refer to `batch_level_label_inference/README.md`

### 2.2. replacement_backdoor_binary
1. the training process of CoAE (in folder `train_CoAE`)
2. the main task accuracy vs. replacement backdoor task accuracy under various defense strategies
3. More detail information please refer to `replacement_backdoor_binary/README.md`

### 2.3. replacement_backdoor_multi
0. (the same attack and defenses as `replacement_backdoor_binary` but done by 3 unlabeled parties together)
1. the training process of CoAE (in folder `train_CoAE`)
2. the main task accuracy vs. replacement backdoor task accuracy under various defense strategies
3. More detail information please refer to `replacement_backdoor_multi/README.md`