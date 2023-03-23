#!/bin/bash

# default values
VALUE_T="result/temp"  # path to the folder to save the final results
VALUE_U="result"       # path to the folder to save the temporary results.
VALUE_Z=5              # number of trials.
VALUE_P=5              # number of parallelly executed experiments.
VALUE_D="breastcancer" # name of dataset.
VALUE_N=-1             # number of data records sampled for training.
VALUE_F=0.5            # ratio of features owned by the active party.
VALUE_V=-1             # ratio of features owned by the passive party. if v=-1, the ratio of local features will be 1 - f.
VALUE_I=-1             # setting of feature importance. -1: normal, 1: unbalance
VALUE_M="xgboost"      # type of training algorithm. `r`: Random Forest, `x`: XGBoost, `s`: SecureBoost
VALUE_R=5              # total number of rounds for training.
VALUE_J=1              # minimum number of samples within a leaf.
VALUE_H=6              # maximum depth
VALUE_A=0.3            # learning rate of XGBoost.
VALUE_E=0.6            # coefficient of edge weight (tau in our paper).
VALUE_K=1.0            # weight for community variables.
VALUE_L=100            # maximum number of iterations of Louvain
VALUE_C=0              # number of completely secure rounds.
VALUE_B=-1             # epsilon of ID-LMID.
VALUE_O=-1             # epsilon of LP-MST.

while getopts d:m:r:c:a:h:j:n:f:v:e:l:o:z:t:u:p:b:k:i:xgq OPT; do
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
  "b")
    FLG_B="TRUE"
    VALUE_B="$OPTARG"
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
  "v")
    FLG_V="TRUE"
    VALUE_V="$OPTARG"
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
  "x")
    FLG_X="TRUE"
    VALUE_X="$OPTARG"
    ;;
  "g")
    FLG_G="TRUE"
    VALUE_G="$OPTARG"
    ;;
  "q")
    FLG_Q="TRUE"
    VALUE_Q="$OPTARG"
    ;;
  esac
done

RESUD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=${VALUE_U})
TEMPD=$(mktemp -d -t ci-$(date +%Y-%m-%d-%H-%M-%S)-XXXXXXXXXX --tmpdir=${VALUE_T})

echo -e "d,${VALUE_D}\nm,${VALUE_M}\nr,${VALUE_R}\nc,${VALUE_C}\na,${VALUE_A}\nh,${VALUE_H}\nb,${VALUE_B}\ni,${VALUE_I}\ne,${VALUE_E}\nl,${VALUE_L}\no,${VALUE_O}\nn,${VALUE_N}\nf,${VALUE_F}\nv,${VALUE_V}\nk,${VALUE_K}\nj,${VALUE_J}\nz,${VALUE_Z}\nx,${FLG_X}" >"${RESUD}/param.csv"

if [ "${VALUE_M}" = "xgboost" ] || [ "${VALUE_M}" = "x" ]; then
  cp build/script/train_xgboost build/script/pipeline_1_training.out
elif [ "${VALUE_M}" = "secureboost" ] || [ "${VALUE_M}" = "s" ]; then
  cp build/script/train_secureboost build/script/pipeline_1_training.out
elif [ "${VALUE_M}" = "randomforest" ] || [ "${VALUE_M}" = "r" ]; then
  cp build/script/train_randomforest build/script/pipeline_1_training.out
else
  echo "m=${VALUE_M} is not supported"
fi

for s in $(seq 1 ${VALUE_Z}); do
  TRAINCMD="script/run_training.sh -s ${s} -d ${VALUE_D} -m ${VALUE_M} -p ${TEMPD} -r ${VALUE_R} -c ${VALUE_C} -a ${VALUE_A} -h ${VALUE_H} -b ${VALUE_B} -j ${VALUE_J} -n ${VALUE_N} -f ${VALUE_F} -v ${VALUE_V} -e ${VALUE_E} -l ${VALUE_L} -o ${VALUE_O} -k ${VALUE_K} -i ${VALUE_I}"
  if [ "${FLG_X}" = "TRUE" ]; then
    TRAINCMD+=" -x"
  fi
  if [ "${FLG_G}" = "TRUE" ]; then
    TRAINCMD+=" -g"
  fi
  if [ "${FLG_Q}" = "TRUE" ]; then
    TRAINCMD+=" -q"
  fi
  if [ ${VALUE_P} -gt 1 ]; then
    if [ $((${s} % ${VALUE_P})) -ne 0 ] && [ ${s} -ne ${VALUE_Z} ]; then
      TRAINCMD+=" &"
    else
      TRAINCMD+=" & wait"
    fi
  fi
  eval ${TRAINCMD}
done

script/run_extract_result.sh -o ${TEMPD}

if [ "${FLG_G}" = "TRUE" ]; then
  echo "Drawing a network ..."
  if [ "${FLG_X}" = "TRUE" ]; then
    python3 script/pipeline_3_vis_union.py -p ${TEMPD}
  else
    python3 script/pipeline_3_vis_network.py -p ${TEMPD} -e ${VALUE_E}
  fi
fi

echo "Making a report ..."
python3 script/pipeline_4_report.py -p ${TEMPD} >"${RESUD}/report.md"

mv ${TEMPD}/*.ans ${RESUD}/
mv ${TEMPD}/*.sratio ${RESUD}/
mv ${TEMPD}/leak.csv ${RESUD}/
mv ${TEMPD}/loss_lp.csv ${RESUD}/
mv ${TEMPD}/result.png ${RESUD}/

if [ "${FLG_Q}" = "TRUE" ]; then
  mv ${TEMPD}/*.html ${RESUD}/
fi

for s in $(seq 1 ${VALUE_Z}); do
  if [ -e ${TEMPD}/${s}_adj_mat_plot.png ]; then
    mv ${TEMPD}/${s}_adj_mat_plot.png ${RESUD}/
  fi
  if [ -e ${TEMPD}/${s}_union_plot.png ]; then
    mv ${TEMPD}/${s}_union_plot.png ${RESUD}/
  fi
done

wait
rm -rf ${TEMPD}
