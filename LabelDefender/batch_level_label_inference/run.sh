# run main task evaluation, examples
python vfl_main_task_ae.py --dataset_name mnist --lr 0.001 --epochs 30
python vfl_main_task_ae.py --dataset_name nuswide --lr 0.001 --epochs 20
python vfl_main_task_ae.py --dataset_name cifar10 --lr 0.003 --epochs 100
python vfl_main_task_ae.py --dataset_name cifar100 --lr 0.003 --epochs 200
python vfl_main_task_gaussian.py --dataset_name cifar100 --lr 0.003 --epochs 200
python vfl_main_task_grad_spars.py --dataset_name cifar100 --lr 0.003 --epochs 200
python vfl_main_task_laplace.py --dataset_name cifar100 --lr 0.003 --epochs 200
python vfl_main_task_marvell.py --dataset_name cifar100 --lr 0.003 --epochs 200
python vfl_main_task_model_completion.py --dataset_name cifar100 --apply_ppdl True --ppdl_theta_u 0.75 --lr 0.003 --epochs 200
python vfl_main_task_model_completion.py --dataset_name cifar100 --apply_gc True --gc_preserved_percent 0.1 --lr 0.003 --epochs 200
python vfl_main_task_model_completion.py --dataset_name cifar100 --apply_lap_noise True --noise_scale 1e-3 --lr 0.003 --epochs 200
python vfl_main_task_model_completion.py --dataset_name cifar100 --apply_discrete_gradients True --discrete_gradients_bins 12 --lr 0.003 --epochs 200
python vfl_main_task_no_defense.py --dataset_name cifar100 --lr 0.003 --epochs 200


# run attack task evaluation, examples
python vfl_dlg_ae.py --dataset mnist
python vfl_dlg_ae.py --dataset nuswide
python vfl_dlg_ae.py --dataset cifar10
python vfl_dlg_ae.py --dataset cifar100
python vfl_dlg_gaussian.py --dataset cifar100 
python vfl_dlg_grad_spars.py --dataset cifar100
python vfl_dlg_laplace.py --dataset cifar100
python vfl_dlg_marvell.py --dataset cifar100
python vfl_dlg_model_completion.py --dataset cifar100 --apply_ppdl True --ppdl_theta_u 0.75
python vfl_dlg_model_completion.py --dataset cifar100 --apply_gc True --gc_preserved_percent 0.1
python vfl_dlg_model_completion.py --dataset cifar100 --apply_lap_noise True --noise_scale 1e-3
python vfl_dlg_model_completion.py --dataset cifar100 --apply_discrete_gradients True --discrete_gradients_bins 12
python vfl_dlg_no_defense.py --dataset cifar100

last_exit_status=$?
echo echo $last_exit_status