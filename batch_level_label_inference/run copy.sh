# run main task evaluation, examples
python vfl_main_task_model_completion.py --dataset_name mnist --apply_lap_noise True --noise_scale 1e-3 --lr 0.003 --epochs 1
python vfl_main_task_model_completion.py --dataset_name mnist --apply_discrete_gradients True --discrete_gradients_bins 12 --lr 0.003 --epochs 1
python vfl_main_task_no_defense.py --dataset_name mnist --lr 0.003 --epochs 1


# run attack task evaluation, examples
python vfl_dlg_ae.py --dataset mnist
python vfl_dlg_gaussian.py --dataset mnist 
python vfl_dlg_grad_spars.py --dataset mnist
python vfl_dlg_laplace.py --dataset mnist
python vfl_dlg_marvell.py --dataset mnist
python vfl_dlg_model_completion.py --dataset mnist --apply_ppdl True --ppdl_theta_u 0.75
python vfl_dlg_model_completion.py --dataset mnist --apply_gc True --gc_preserved_percent 0.1
python vfl_dlg_model_completion.py --dataset mnist --apply_lap_noise True --noise_scale 1e-3
python vfl_dlg_model_completion.py --dataset mnist --apply_discrete_gradients True --discrete_gradients_bins 12
python vfl_dlg_no_defense.py --dataset mnist

last_exit_status=$?
echo echo $last_exit_status