#! /bin/bash

for i in `seq 0 10`; do
    # for attack only, without defense
    python main.py --name attack --dataset mnist --model mlp2 --seed $i --epoch 100 --backdoor 1 --defense_up 0 --learning_rate 0.01 --gpu 0
    python main.py --name attack --dataset cifar10 --model resnet18 --seed $i --epoch 100 --backdoor 1 --defense_up 0 --learning_rate 0.01 --gpu 0
    python main.py --name attack --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --defense_up 0 --learning_rate 0.01 --gpu 0
    # defense
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --defense_up 1 --learning_rate 0.01 --gpu 0 --dp_type laplace --dp_strength 0.001
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --defense_up 1 --learning_rate 0.01 --gpu 0 --dp_type gaussian --dp_strength 0.005
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --defense_up 1 --learning_rate 0.01 --gpu 0 --gradient_sparsification 99.0
    python main_replace_per_comm.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --defense_up 1 --learning_rate 0.01 --gpu 0 --autoencoder 1 --lba 1.0 --model_timestamp 1645374585
    python main_replace_per_comm.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --defense_up 1 --learning_rate 0.01 --gpu 0 --autoencoder 1 --lba 1.0 --model_timestamp 1645374585 --apply_discrete_gradients True --discrete_gradients_bins 12
done

last_exit_status=$?
echo echo $last_exit_status