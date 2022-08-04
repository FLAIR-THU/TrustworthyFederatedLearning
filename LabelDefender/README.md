# Label Defender

> This folder is the pytorch implementation for [Defending Batch-Level Label Inference and Replacement Attacks in Vertical Federated Learning](https://www.computer.org/csdl/journal/bd/5555/01/09833321/1F8uKhxrvNe).
> For citation, please copy and paste the below into `.bib` file.
```
@ARTICLE {9833321,
author = {T. Zou and Y. Liu and Y. Kang and W. Liu and Y. He and Z. Yi and Q. Yang and Y. Zhang},
journal = {IEEE Transactions on Big Data},
title = {Defending Batch-Level Label Inference and Replacement Attacks in Vertical Federated Learning},
year = {2022},
volume = {},
number = {01},
issn = {2332-7790},
pages = {1-12},
doi = {10.1109/TBDATA.2022.3192121},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {jul}
}
```

## 1. Attacks and Defenses
* In the [paper](https://www.computer.org/csdl/journal/bd/5555/01/09833321/1F8uKhxrvNe), we proposed 2 novel attacks at passive part (party without label information), namely, `Batch-level Label Inference Attack` and `Label Replacement Backdoor Attack`. These two attacks are implemented separately in folder `batch_level_label_inference` and `replacement_backdoor_binary`. In folder `replacement_backdoor_multi`, we extend our backdoor attack experiment from VFL setting with 2 parties into VFL setting with 4 parties. 

* We also proposed 2 novel defending methods, namely `Confusional AutoEncoder (CAE)` and `DiscreteSGD-enhanced CAE (DCAE)`. The defense method is implemented in every folder for defending against the above attacks.

* For quick start of running our code, please refer to the `README.md` file in each folder.

## 2. Folders List
| Folder | Setting| Attack | Defense | Paper|
|:-:|:-:|:-:|:-:|:-:|
|batch_level_label_inference| VFL, 2-party | Batch-level Label Inference by gradient inversion | Confusional AutoEncoder([CAE](https://arxiv.org/abs/2112.05409)), Disceret-SGD enchanced CAE([DCAE](https://arxiv.org/abs/2112.05409)), baseline defenses | [link](https://arxiv.org/abs/2112.05409) Section 4 |
|replacement_backdoor_binary| VFL, 2-party | Replacement Backdoor by replacing labels | Confusional AutoEncoder([CAE](https://arxiv.org/abs/2112.05409)), Disceret-SGD enchanced CAE([DCAE](https://arxiv.org/abs/2112.05409)), baseline defenses | [link](https://arxiv.org/abs/2112.05409) Section 5 |
|replacement_backdoor_multi| VFL, 4-party | Replacement Backdoor by replacing labels | Confusional AutoEncoder([CAE](https://arxiv.org/abs/2112.05409)), Disceret-SGD enchanced CAE([DCAE](https://arxiv.org/abs/2112.05409)), baseline defenses | [link](https://arxiv.org/abs/2112.05409) Section 5 |

* baseline defenses includes: Differencial Privacy, Gradient Sparsification, Marvell, Priviacy Preserving Deep Learning, Discrete-SGD.


## 3. Folder Contents

### 3.1. batch_level_label_inference
1. the training process of CoAE (in folder `train_CoAE`)
2. the trained models of CAE (in folder `trained_models`)
3. the main task accuracy vs. label inference task accuracy under various defense strategies
4. More detail information please refer to `batch_level_label_inference/README.md`

### 3.2. replacement_backdoor_binary
1. the training process of CoAE (in folder `train_CoAE`)
2. the main task accuracy vs. replacement backdoor task accuracy under various defense strategies
3. More detail information please refer to `replacement_backdoor_binary/README.md`

### 3.3. replacement_backdoor_multi
0. (the same attack and defenses as `replacement_backdoor_binary` but done by 3 unlabeled parties together)
1. the training process of CoAE (in folder `train_CoAE`)
2. the main task accuracy vs. replacement backdoor task accuracy under various defense strategies
3. More detail information please refer to `replacement_backdoor_multi/README.md`