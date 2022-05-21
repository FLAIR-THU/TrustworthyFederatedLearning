# vfl_backdoor_pytorch

Settings:


| Dataset      | Model |
| ----------- | ----------- |
| mnist      | mlp2       |
| nuswide   | mlp2        |
| cifar10   | resnet18    |
| cifar20   | resnet18    |
| cifar100   | resnet18    |


### Usage:

```
python main.py --dataset mnist --model mlp2 --backdoor 1 --amplify_rate 10 --seed XX
```

### Defense:

#### DP

Possible dp types: ['laplace', 'gaussian']

Possible strength values: [0.1, 0.01, 0.001]

```
python main.py --dataset mnist --model mlp2 --backdoor 1 --amplify_rate 10 --seed XX \
--dp_type gaussian --dp_strength 0.01

```
#### Gradient sparsification
Possible values: [99, 99.5, 99.9]
```
python main.py --dataset mnist --model mlp2 --backdoor 1 --amplify_rate 10 --seed XX \
--gradient_sparsification 99

```

Default data directory should be: ` ../dataset/target_datapth`