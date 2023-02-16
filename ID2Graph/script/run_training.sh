while getopts d:m:p:n:f:v:r:c:a:h:b:j:e:l:o:z:k:s:i:xgq OPT; do
  case $OPT in
  "d")
    FLG_D="TRUE"
    VALUE_D="$OPTARG"
    ;;
  "m")
    FLG_M="TRUE"
    VALUE_M="$OPTARG"
    ;;
  "p")
    FLG_P="TRUE"
    VALUE_P="$OPTARG"
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
  "s")
    FLG_S="TRUE"
    VALUE_S="$OPTARG"
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

PREPCMD="python3 ./data/prep.py -d ${VALUE_D} -p ./data/${VALUE_D}/ -n ${VALUE_N} -f ${VALUE_F} -v ${VALUE_V} -s ${VALUE_S} -i ${VALUE_I}"
eval ${PREPCMD}

cp "./data/${VALUE_D}/${VALUE_D}_${VALUE_S}.in" "${VALUE_P}/${VALUE_S}_data.in"

RUNCMD="build/script/pipeline_1_training.out -f ${VALUE_P} -p ${VALUE_S} -r ${VALUE_R} -h ${VALUE_H} -b ${VALUE_B} -j ${VALUE_J} -c ${VALUE_C} -e ${VALUE_E} -l ${VALUE_L} -o ${VALUE_O}"
if [ "${VALUE_M}" = "xgboost" ] || [ "${VALUE_M}" = "x" ] || [ "${VALUE_M}" = "secureboost" ] || [ "${VALUE_M}" = "s" ]; then
  RUNCMD+=" -a ${VALUE_A}"
fi
if [ "${FLG_G}" = "TRUE" ]; then
  RUNCMD+=" -g"
fi
if [ "${FLG_Q}" = "TRUE" ]; then
  RUNCMD+=" -q"
fi
if [ "${FLG_X}" = "TRUE" ]; then
  RUNCMD+=" -x"
fi
eval ${RUNCMD} <"${VALUE_P}/${VALUE_S}_data.in"

if [ "${FLG_X}" = "TRUE" ]; then
  echo "Start Union Tree Attack trial=${VALUE_S}"
  python3 script/pipeline_2_uniontree.py -p "${VALUE_P}/${VALUE_S}_data.in" -q "${VALUE_P}/${VALUE_S}_union.out" -s ${VALUE_S} >"${VALUE_P}/${VALUE_S}_leak.csv"
else
  echo "Start Clustering trial=${VALUE_S}"
  CLSCMD="python3 script/pipeline_2_clustering.py -p ${VALUE_P}/${VALUE_S}_data.in -q ${VALUE_P}/${VALUE_S}_communities.out -k ${VALUE_K} -s ${VALUE_S}"
  eval ${CLSCMD} >"${VALUE_P}/${VALUE_S}_leak.csv"
  echo "Clustering is complete trial=${VALUE_S}"
fi
