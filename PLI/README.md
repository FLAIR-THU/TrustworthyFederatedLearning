# Breaching FedMD: Image Recovery via Paired-Logits Inversion Attack

Implementation of **Breaching FedMD: Image Recovery via Paired-Logits Inversion Attack** (CVPR 2023)

<img src="img/overview.drawio.svg">

## Install

```
./install.sh
```

## Datasets

We assume that all data locates in `data` folder.

```
├── data
    ├── LAG
    │    ├── 50_cent
    │    │      .
    │    │      .
    │
    ├── LFW
    |    ├── lfw-align-128
    |    |      ├── AJ_Cook
    |    |      ├──    .
    |    |      ├──    .
    |    |
    |    └── lfw-align-128-masked
    |           ├── AJ_Cook
    |           ├──    .
    |           ├──    .
    |
    └── FaceScrub
            ├── actors_faces
            └── actresses_face
```

## Usage

- Example

```
python script/main.py -t FedMD -d LAG -a pli -p ./data/lag -o path_to_output_folder
```

- Basic Optional Arguments

For more details, please check [script/main.py](script/main.py).

```
usage: main.py [-h] [-t FEDKD_TYPE] [-d DATASET] [-a ATTACK_TYPE] [-c CLIENT_NUM] [-s SOFTMAX_TEMPREATURE]
               [-p PATH_TO_DATAFOLDER] [-o OUTPUT_FOLDER] [-b ABLATION_STUDY]

optional arguments:
  -h, --help            show this help message and exit
  -t FEDKD_TYPE, --fedkd_type FEDKD_TYPE
                        type of FedKD;
                            FedMD, FedGEMS, or FedGEMS
  -d DATASET, --dataset DATASET
                        type of dataset;
                            LAG, LFW, or FaceScrub
  -a ATTACK_TYPE, --attack_type ATTACK_TYPE
                        type of attack;
                            pli or tbi
  -c CLIENT_NUM, --client_num CLIENT_NUM
                        number of clients
  -s SOFTMAX_TEMPREATURE, --softmax_tempreature SOFTMAX_TEMPREATURE
                        tempreature $ au$
  -p PATH_TO_DATAFOLDER, --path_to_datafolder PATH_TO_DATAFOLDER
                        path to the data folder
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        path to the output folder
  -b ABLATION_STUDY, --ablation_study ABLATION_STUDY
                        type of ablation study;
                     0: only local logits with prior-based inference adjusting
                     1: only local logits witout inference adjusting
                     2: paird logits with prior-based inference adjusting (default)
```

## Citation

```
@inproceedings{takahashi2021breaching,
  title={Breaching FedMD: Image Recovery via Paired-Logits Inversion Attack},
  author={Takahashi, H and Liu, J and Liu, Y and Liu, Y},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
