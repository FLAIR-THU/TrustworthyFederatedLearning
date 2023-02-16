while getopts o: OPT; do
  case $OPT in
  "o")
    FLG_O="TRUE"
    VALUE_O="$OPTARG"
    ;;
  esac
done

ROUND_NUM=$(cat ${VALUE_O}/1_result.ans | grep -oP '(?<=round)(.*?)(?=:)' | tail -n1)

for i in $(seq 1 ${ROUND_NUM}); do
  cat ${VALUE_O}/*_result.ans | grep -oP "(?<=Tree-${i}: )(.*)" >"${VALUE_O}/temp_lp_tree_${i}.out"
  cat ${VALUE_O}/*_result.ans | grep -oP "(?<=round ${i}: )(.*)" >"${VALUE_O}/temp_loss_tree_${i}.out"
done

cat ${VALUE_O}/*_result.ans | grep -oP '(?<=Train AUC,)(.*)' >"${VALUE_O}/temp_train_auc.out"
cat ${VALUE_O}/*_result.ans | grep -oP '(?<=Val AUC,)(.*)' >"${VALUE_O}/temp_val_auc.out"
