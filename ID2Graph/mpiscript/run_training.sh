while getopts d:m:p:n:f:i:r:c:a:h:e:s: OPT; do
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
    "e")
        FLG_E="TRUE"
        VALUE_E="$OPTARG"
        ;;
    "s")
        FLG_S="TRUE"
        VALUE_S="$OPTARG"
        ;;
    esac
done

echo "random seed is ${VALUE_S}"
echo ${VALUE_N}
python3 ./data/prep.py -d ${VALUE_D} -p "./data/${VALUE_D}/" -n ${VALUE_N} -f ${VALUE_F} -i ${VALUE_I} -s ${VALUE_S}
cp "./data/${VALUE_D}/${VALUE_D}_${VALUE_S}.in" "${VALUE_P}/${VALUE_S}_0_data.in"
cp "./data/${VALUE_D}/${VALUE_D}_${VALUE_S}.in" "${VALUE_P}/${VALUE_S}_1_data.in"

RUNCMD="mpirun -np 2 build/mpiscript/pipeline_1_training.out -f ${VALUE_P} -p ${VALUE_S} -r ${VALUE_R} -h ${VALUE_H} -c ${VALUE_C} -e ${VALUE_E}"
if [ "${VALUE_M}" = "secureboost" ] || [ "${VALUE_M}" = "s" ]; then
    RUNCMD+=" -a ${VALUE_A}"
fi

eval ${RUNCMD} #<"${VALUE_P}/${VALUE_S}_data.in"
