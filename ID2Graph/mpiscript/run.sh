#!/bin/bash

# constant values
NUM_TRIAL=5

# default values
VALUE_D="breastcancer"
VALUE_M="secureboost"
VALUE_R=20
VALUE_C=1
VALUE_A=0.3
VALUE_H=3
VALUE_J=1
VALUE_N=20000
VALUE_F=0.5
VALUE_I=1
VALUE_E=0.3
VALUE_K="vanila"
VALUE_T="result/temp"
VALUE_U="result"
VALUE_P=1
VALUE_L=0.0
VALUE_Z=300
VALUE_O=-1

while getopts d:m:r:c:a:h:j:n:f:i:e:l:o:z:t:u:p:wg OPT; do
    case $OPT in
    "d")
        FLG_D="TRUE"
        VALUE_D="$OPTARG"
        ;;
    "m")
        FLG_M="TRUE"
        VALUE_M="$OPTARG"
        ;;
    "r")
        FLG_R="TRUE"
        VALUE_R="$OPTARG"
        ;;
    "c")
        FLG_C="TRUE"
        VALUE_C="$OPTARG"
        ;;
    "a")
        FLG_A="TRUE"
        VALUE_A="$OPTARG"
        ;;
    "h")
        FLG_H="TRUE"
        VALUE_H="$OPTARG"
        ;;
    "j")
        FLG_J="TRUE"
        VALUE_J="$OPTARG"
        ;;
    "n")
        FLG_N="TRUE"
        VALUE_N="$OPTARG"
        ;;
    "f")
        FLG_F="TRUE"
        VALUE_F="$OPTARG"
        ;;
    "i")
        FLG_I="TRUE"
        VALUE_I="$OPTARG"
        ;;
    "e")
        FLG_E="TRUE"
        VALUE_E="$OPTARG"
        ;;
    "l")
        FLG_L="TRUE"
        VALUE_L="$OPTARG"
        ;;
    "o")
        FLG_O="TRUE"
        VALUE_O="$OPTARG"
        ;;
    "z")
        FLG_Z="TRUE"
        VALUE_Z="$OPTARG"
        ;;
    "k")
        FLG_K="TRUE"
        VALUE_K="$OPTARG"
        ;;
    "t")
        FLG_T="TRUE"
        VALUE_T="$OPTARG"
        ;;
    "u")
        FLG_U="TRUE"
        VALUE_U="$OPTARG"
        ;;
    "p")
        FLG_P="TRUE"
        VALUE_P="$OPTARG"
        ;;
    "w")
        FLG_W="TRUE"
        VALUE_W="$OPTARG"
        ;;
    "g")
        FLG_G="TRUE"
        VALUE_G="$OPTARG"
        ;;
    esac
done

RESUD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=${VALUE_U})
TEMPD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=${VALUE_T})

echo -e "d,${VALUE_D}\nm,${VALUE_M}\nr,${VALUE_R}\nc,${VALUE_C}\na,${VALUE_A}\nh,${VALUE_H}\ni,${VALUE_I}\ne,${VALUE_E}\nl,${VALUE_L}\no,${VALUE_O}\nw,${FLG_W}\nn,${VALUE_N}\nf,${VALUE_F}\nk,${VALUE_K}" >"${RESUD}/param.csv"

if [ "${VALUE_M}" = "secureboost" ] || [ "${VALUE_M}" = "s" ]; then
    cp build/mpiscript/train_mpisecureboost build/mpiscript/pipeline_1_training.out
elif [ "${VALUE_M}" = "federatedforest" ] || [ "${VALUE_M}" = "f" ]; then
    cp build/mpiscript/train_mpirandomforest build/mpiscript/pipeline_1_training.out
else
    echo "m=${VALUE_M} is not supported"
fi

for s in $(seq 1 ${NUM_TRIAL}); do
    mpiscript/run_training.sh -s ${s} -d ${VALUE_D} -m ${VALUE_M} -p ${TEMPD} -r ${VALUE_R} -c ${VALUE_C} -a ${VALUE_A} -h ${VALUE_H} -n ${VALUE_N} -f ${VALUE_F} -i ${VALUE_I} -e ${VALUE_E}
done

script/run_extract_result.sh -o ${TEMPD}

mv ${TEMPD}/*.ans ${RESUD}/

wait
rm -rf ${TEMPD}
