#! /bin/bash
# author zty
# "for" for ploting


python train_autoencoder.py --nClasses 5 --lba 2.0
python train_autoencoder.py --nClasses 5 --lba 1.0
python train_autoencoder.py --nClasses 5 --lba 0.5
python train_autoencoder.py --nClasses 5 --lba 0.1
python train_autoencoder.py --nClasses 5 --lba 0.0

python train_autoencoder.py --nClasses 20 --lba 2.0
python train_autoencoder.py --nClasses 20 --lba 1.0
python train_autoencoder.py --nClasses 20 --lba 0.5
python train_autoencoder.py --nClasses 20 --lba 0.1
python train_autoencoder.py --nClasses 20 --lba 0.0
