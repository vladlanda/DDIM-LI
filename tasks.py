from invoke import task

@task
def inf_h1_dim128_gt(c):
    cmd = (
        f'python infer.py '
        f'--checkpoint  "/media/vladlanda/T5EVO/Postdoc/Tel Aviv/Code/DDIM-LI/outputs/run_1h_dim128/best.pt" '
        f'--data_dir "/home/vladlanda/Workplace/LI-DATASETS/inference/central_africa_4" '
        f'--output_dir  "/media/vladlanda/T5EVO/Postdoc/Tel Aviv/Code/DDIM-LI/outputs/inference" ' 
        f'--n_members 32 '
        f'--cfg_scale 1.5 '
        f'--plot '
        f'--img_size 128 128 '
        f'--n_forecasts 1 '
        f'--use_gt '
    )
    c.run(cmd,pty=True)


@task
def eval_h1_dim128(c):

    cmd = (
        f'python evaluate.py '
        f'--checkpoint  "/media/vladlanda/T5EVO/Postdoc/Tel Aviv/Code/DDIM-LI/outputs/run_1h_dim128/best.pt" '
        f'--test_roots "/home/vladlanda/Workplace/LI-DATASETS/inference/central_africa_4" '
        f'--output_dir  "/media/vladlanda/T5EVO/Postdoc/Tel Aviv/Code/DDIM-LI/outputs/evaluation" ' 
        f'--n_members 6 '
        f'--cfg_scale 1 '
        f'--plot --max_plots 3 '
        f'--img_size 128 128 '
    )
    c.run(cmd,pty=True)