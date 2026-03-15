from invoke import task

@task
def inf_h1_dim128(c):
    cmd = (
        f'python infer.py '
        f'--checkpoint  "/media/vladlanda/T5EVO/Postdoc/Tel Aviv/Code/DDIM-LI/outputs/run_1h_dim128/best.pt" '
        f'--context_dir "/home/vladlanda/Workplace/LI-DATASETS/inference/central_africa_4" '
        f'--output_dir  "/media/vladlanda/T5EVO/Postdoc/Tel Aviv/Code/DDIM-LI/outputs/inference" ' 
        f'--n_members 6 '
        f'--cfg_scale 1.5 '
        f'--plot '
        f'--img_size 128 128 '
        f'--n_forecasts 1'
    )
    c.run(cmd,pty=True)
