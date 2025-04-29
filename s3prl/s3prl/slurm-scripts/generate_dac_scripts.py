"""
Run adapter-tuning for hubert for DAC with various hyperparameters
"""

from glob import glob
import numpy as np
import os
import pandas as pd

def get_header(task_name):
    write_str = []
    write_str.append("#! /bin/bash")
    write_str.append("#SBATCH --partition=speech-gpu")
    write_str.append("#SBATCH --gpus=1")
    if task_name == 'slurp':
        write_str.append("#SBATCH --cpus-per-task=6")
        write_str.append("#SBATCH --constraint=12g")
        write_str.append("#SBATCH --array=0-2%1")
    else:
        write_str.append("#SBATCH --cpus-per-task=3")
        write_str.append("#SBATCH --array=0-1%1")

    write_str.append('eval \"$(/share/data/speech/ankitap/miniconda3/bin/conda \"shell.bash\" \"hook\")\"')
    write_str.append("conda activate s3-adapt")

    write_str_header = "\n".join(write_str)
    return write_str_header

def check_status(task_name, fname):
    log_fn = os.path.join(f"../result/downstream/{task_name}-exp", fname, "log.log")
    # if 'superb-u-hubert_cp_lr-0.001_classname-scenario_adapter-lora-16dim_wup-1000_noproj_layer-12' in fname:
    #     import pdb; pdb.set_trace()
    if task_name == "slue":
        tot_steps = "20000"
    elif 'ctc' in task_name or 'dac' in task_name:
        tot_steps = "100000"
    else:
        tot_steps = "200000"
    # import pdb; pdb.set_trace()
    if os.path.exists(log_fn):
        with open(log_fn, "r") as f:
            log_data = [line.strip() for line in f.readlines()]
        if tot_steps in log_data[-1]:
        # if f"test at step {tot_steps}" in log_data[-1]:
            return False
        if 'dac' in task_name: # check if macro-f1 is 0
            for line in log_data[::-1]:
                if 'macro' in line:
                    score = line.split(': ')[-1]
                    if score == '0.0':
                        return False
                    else:
                        return True
        if task_name in ['ctc', 'asr']:
            if 'test' in log_data[-1] and 'step 0' in log_data[-1]:
                return False
    else:
        pass
        # csv_dir = '/share/data/lang/users/ankitap/ap-rep/git_check/layerwise-analysis/packages/forked_s3prl/s3prl/s3prl/result'
        # csv_fn = 'dac_lora_scores'
        # thresh = 73
        # if 'lora' not in fname:
        #     assert 'houlsby' in fname
        #     csv_fn = csv_fn.replace('lora', 'houlsby')
        # if 'dac' not in task_name:
        #     assert 'slurp' in task_name
        #     if 'scenario' in fname:
        #         csv_fn = csv_fn.replace('dac', 'slurp_scenario')
        #         thresh = 75
        #     else:
        #         assert 'action' in fname
        #         csv_fn = csv_fn.replace('dac', 'slurp_action')
        # score_fns = glob(os.path.join(csv_dir, f'*{csv_fn}*'))
        # for fn in score_fns:
        #     # Load the CSV file
        #     data = pd.read_csv(fn)
        
        #     # Check if the exp_name exists in the file and has a macro F1 (dev) score less than 63
        #     if fname in data['exp_name'].values:
        #         exp_data = data[data['exp_name'] == fname]
        #         if exp_data['macro F1 (dev)'].values[0] < thresh:
        #             return False
    return True

def generate_cmd(
        write_str_header,
        task_name,
        model_name,
        sfx,
        adapter,
        lr,
        ln,
        use_proj,
        hdim,
        wup_steps,
        layer,
        slurm_cmd
        ):
    cfg_str = f'config.optimizer.lr={lr}'
    if task_name == "slurp":
        write_fn = f"superb-u-{model_name}_lr-{lr}_classname-{sfx}_adapter-{adapter}-{hdim}dim"
        cfg_str += f",,config.downstream_expert.datarc.classname={sfx}"
    else:
        write_fn = f"superb-u-{model_name}_lr-{lr}_adapter-{adapter}-{hdim}dim"
        cfg_str += ',,config.downstream_expert.objrc.pos_wt=None'
    if wup_steps == 0:
        cfg_fn = f'downstream/{task_name}/config.yaml'
    else:
        cfg_fn = f'downstream/{task_name}/config_schd.yaml'
        cfg_str += f',,config.scheduler.num_warmup_steps={wup_steps}'
        write_fn += f'_wup-{wup_steps}'
    if not use_proj:
        write_fn += '_noproj'
        cfg_str += ',,config.downstream_expert.modelrc.use_proj=False'
    else:
        cfg_str += ',,config.downstream_expert.modelrc.use_proj=True'
    if ln:
        write_fn += '_lnorm'
    if layer != 'all':
        write_fn += f'_layer-{layer}'
    if adapter == 'lora':
        run_fn = 'run_downstream_cp'
    else:
        run_fn = 'run_downstream'
    # import pdb; pdb.set_trace()
    run_flag = check_status(task_name, write_fn)
    if run_flag:
        cmd_str = [write_str_header]
        cmd_str += [f"for i in $(seq 1 10);\n\tdo python3 {run_fn}.py --adapter {adapter} -m train -d {task_name} -a -f \\"]
        cmd_str += [f'-u {model_name} \\']
        cmd_str += [f'-o {cfg_str} \\']
        cmd_str += [f'-c {cfg_fn} \\']
        if adapter == 'houlsby':
            cmd_str += [f'--adapter_dim {hdim} \\']
            if ln:
                cmd_str += ['--houlsby_ln True \\']
            else:
                cmd_str += ['--houlsby_ln False \\']
        else:
            cmd_str += [f'--lora_dim {hdim} \\']
        cmd_str += [f'--config {cfg_fn} \\']
        if layer != 'all':
            cmd_str += [f'-l {layer} \\']
        cmd_str += [f'-n ./{task_name}-exp/{write_fn};']
        cmd_str += ['done']
        with open(os.path.join(task_name, f"{write_fn}.sh"), "w") as f:
            f.write("\n".join(cmd_str))
        slurm_cmd.append(f"sbatch slurm-scripts/{task_name}/{write_fn}.sh")

def write_cmd(save_fn, slurm_cmd, tot_runs):
    with open(f'../{save_fn}', 'w') as f:
        f.write("\n".join(slurm_cmd))
    print(f"{len(slurm_cmd)} of {tot_runs} written to {save_fn}")

def run_sweep(check_log=False):
    # task_lst = ['dac', 'slurp']
    task_lst = ['slurp']
    # task_lst = ['dac']
    adapter_lst = ['lora']
    # adapter_lst = ['lora', 'houlsby']

    # adapter = "houlsby"
    # adapter = "lora"

    # proj_lst = [True, False]
    proj_lst = [False]
    wup_step_lst = [0, 500, 1000, 3000]
    log_write_str = []
    for task_name in task_lst:
        write_str_header = get_header(task_name)
        if 'dac' in task_name:
            task_name = 'slue_hvb_dac'
        for adapter in adapter_lst:
            slurm_cmd = []
            tot_runs = 0
            if adapter == 'lora':
                model_name = "hubert_cp"
                ln_lst = [False]
                hdim_lst = [8, 16, 32, 64]
                if 'dac' in task_name:
                    lr_lst = [1e-3]
                else:
                    lr_lst = [1e-3]
                    # lr_lst = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
                layer_lst = [12]
            else:
                model_name = "hubert"
                ln_lst = [True, False]
                hdim_lst = [32, 64, 128, 256]
                if 'dac' in task_name:
                    lr_lst = [1e-1, 1e-2, 1e-3]
                else:
                    lr_lst = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
                layer_lst = ['all', 12]
            for lr in lr_lst:
                for ln in ln_lst:
                    for use_proj in proj_lst:
                        for hdim in hdim_lst:
                            for wup_steps in wup_step_lst:
                                for layer in layer_lst:
                                    if task_name == "slurp":
                                        sfx_lst = ["action"]
                                        # sfx_lst = ["scenario"]#, "action"]
                                    else:
                                        sfx_lst = [""]
                                    for sfx in sfx_lst:
                                        tot_runs += 1
                                        os.makedirs(task_name, exist_ok=True)
                                        generate_cmd(
                                            write_str_header,
                                            task_name,
                                            model_name,
                                            sfx,
                                            adapter,
                                            lr,
                                            ln,
                                            use_proj,
                                            hdim,
                                            wup_steps,
                                            layer,
                                            slurm_cmd
                                            )
    
            if check_log:
                with open("output.csv", "w") as f:
                    f.write("\n".join(log_write_str))
            else:
                write_cmd(f'exe_{task_name}_{adapter}.sh', slurm_cmd, tot_runs)

run_sweep()



