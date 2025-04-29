"""
Run adapter-tuning for hubert for DAC with various hyperparameters
"""

import itertools
from glob import glob
import numpy as np
import os
import pandas as pd
import shutil

import torch


def get_header(task_name):
    write_str = []
    write_str.append("#! /bin/bash")
    write_str.append("#SBATCH --partition=speech-gpu")
    write_str.append("#SBATCH --gpus=1")
    if task_name == 'slurp':
        write_str.append("#SBATCH --cpus-per-task=6")
        write_str.append("#SBATCH --constraint=12g")
        write_str.append("#SBATCH --array=0-3%1")
    else:
        write_str.append("#SBATCH --cpus-per-task=3")
        write_str.append("#SBATCH --array=0-1%1")

    write_str.append('eval \"$(/share/data/speech/ankitap/miniconda3/bin/conda \"shell.bash\" \"hook\")\"')
    write_str.append("conda activate s3-adapt")

    if 'slue' in task_name:
        write_str.append('if [ ! -d "/scratch/ankitap/slue-hvb" ]; then')
        write_str.append('mkdir -p /scratch/ankitap')
        write_str.append('cp -r /share/data/speech/hackathon_2022/data/slue_hvb/slue-hvb /scratch/ankitap/')
        write_str.append('fi')
    elif 'slurp' in task_name:
        write_str.append('if [ ! -d "/scratch/ankitap/slurp" ]; then')
        write_str.append('mkdir -p /scratch/ankitap')
        write_str.append('cp -r /share/data/speech/hackathon_2022/data/slurp  /scratch/ankitap/')
        write_str.append('fi')
    write_str_header = "\n".join(write_str)
    return write_str_header

def check_and_fix_ckpt(log_fn):
    exp_dir = '/'.join(log_fn.split('/')[:-1])
    ckpt_fn = glob(os.path.join(exp_dir, 'states-*.ckpt'))
    assert len(ckpt_fn) < 2
    if len(ckpt_fn) == 1:
        ckpt_fn = ckpt_fn[0]
        try:
            # Try loading checkpoint to make sure that the file is not corrupt
            ckpt = torch.load(ckpt_fn, map_location='cpu')
        except:
            # remove faulty checkpoint
            os.remove(ckpt_fn)
            # make a copy of the correct checkpoint
            if 'dac' in log_fn:
                ckpt_fn = os.path.join(exp_dir, 'dev-best-macro-f1.ckpt')
            elif 'slurp' in log_fn:
                ckpt_fn = os.path.join(exp_dir, 'dev-best-acc.ckpt')
            if os.path.join(ckpt_fn):
                ckpt = torch.load(ckpt_fn, map_location='cpu')
                step_num = ckpt['Step']
                shutil.copy(ckpt_fn, os.path.join(exp_dir, f'states-{step_num}.ckpt'))

def check_status(task_name, fname):
    log_fn = os.path.join(f"../result/downstream/{task_name}-exp", fname, "log.log")
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
                        # check_and_fix_ckpt(log_fn)
                        return True
        if task_name in ['ctc', 'asr']:
            if 'test' in log_data[-1] and 'step 0' in log_data[-1]:
                return False
    else:
        # pass
        csv_dir = '/share/data/lang/users/ankitap/ap-rep/git_check/layerwise-analysis/packages/forked_s3prl/s3prl/s3prl/result'
        csv_fn = 'dac_lora_placement_scores'
        thresh = 73
        break_idx = 3
        metric_name = 'macro F1 (dev)'
        if 'dac' not in task_name:
            assert 'slurp' in task_name
            break_idx = 4
            metric_name = 'ACC (dev)'
            if 'scenario' in fname:
                csv_fn = csv_fn.replace('dac', 'slurp_scenario')
                thresh = 75
            else:
                assert 'action' in fname
                csv_fn = csv_fn.replace('dac', 'slurp_action')
        layer_idx = '-'.join(fname.split('_')[break_idx].split('-')[3].split(','))
        fname = ('_'.join(fname.split('_')[:break_idx]) \
                 + '_' + '-'.join(fname.split('_')[break_idx].split('-')[:3]) \
                    + f'-{layer_idx}_' \
                        + '_'.join(fname.split('_')[break_idx+1:]))
        score_fns = glob(os.path.join(csv_dir, f'*{csv_fn}*'))
        for fn in score_fns:
            # Load the CSV file
            data = pd.read_csv(fn)
        
            # Check if the exp_name exists in the file and has a macro F1 (dev) score less than 63
            if fname in data['exp_name'].values:
                exp_data = data[data['exp_name'] == fname]
                if exp_data[metric_name].values[0] < thresh:
                    return False
    return True

def generate_cmd(task_name, model_name, sfx, adapter, lr, ln, use_proj, hdim, wup_steps, layer, slurm_cmd, write_str_header, lora_layers=None):
    cfg_str = f'config.optimizer.lr={lr}'
    if task_name == "slurp":
        write_fn = f"superb-u-{model_name}_lr-{lr}_classname-{sfx}_adapter-{adapter}-{hdim}dim"
        cfg_str += f",,config.downstream_expert.datarc.classname={sfx}"
    else:
        write_fn = f"superb-u-{model_name}_lr-{lr}_adapter-{adapter}-{hdim}dim"
    if lora_layers:
        lora_layers_str = ','.join(list(map(str, eval(lora_layers))))
        write_fn += f'-{lora_layers_str}'
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
        if lora_layers:
            cmd_str += [f'--peft_layer_lst {lora_layers} \\']
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

def get_layer_combinations(lst, subset_size):
    return [item for item in list(itertools.combinations(lst, subset_size))]

def run_sweep(check_log=False, normalize=False):
    # model_lst = ["wav2vec2", "wav2vec2_large_ll60k", "hubert", "hubert_large_ll60k"]
    # model_lst += ["data2vec", "data2vec_large_l160k"]
    # task_lst = ["asr", "slurp", "fluent_commands", "ctc"]
    # task_lst = ["slue_hvb_dac"]#, "slurp"]
    task_lst = ["slurp"]
    # adapter = "houlsby"
    adapter = "lora"

    if adapter == 'lora':
        model_name = "hubert_cp"
        ln_lst = [False]
        hdim_lst = [8]
        # hdim_lst = [8, 16, 32, 64]
    else:
        model_name = "hubert"
        ln_lst = [True, False]
        hdim_lst = [32, 64, 128, 256]

    # lr_lst = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    lr_lst = [1e-3]
    # proj_lst = [True, False]
    proj_lst = [False]
    layer_lst = [12]
    
    # lora_layer_lst = ['[12]',
    #              '[10,11]',
    #              '[11,12]',
    #              '[1,12]',
    #              '[9,10]',
    #              '[8,9,10,11,12]',
    #              '[9,10,11,12]',
    #              '[10,11,12]',
    #              '[9,10,11]',
    #              '[8,9,10]',
    #              '[8,9,10,11]',
    #              '[1,2,11,12]',
    #              '[1,2,3,10,11,12]']
    # wup_step_lst = [0, 500, 1000, 3000]
    
    lora_layer_lst = []
    for num_layers in [4]:
    # for num_layers in [1, 2, 3, 4]:
        lora_layer_lst += get_layer_combinations(np.arange(1, 13), num_layers)
    lora_layer_lst = [''.join(str(list(item)).split(' ')) for item in lora_layer_lst]
    wup_step_lst = [1000]
    lora_layer_lst += ['[1,2,3,4,5,6]', '[7,8,9,10,11,12]']
    
    slurm_cmd = []
    log_write_str = []
    tot_runs = 0
    for task_name in task_lst:
        write_str_header = get_header(task_name)
        for lr in lr_lst:
            for ln in ln_lst:
                for use_proj in proj_lst:
                    for hdim in hdim_lst:
                        for wup_steps in wup_step_lst:
                            for layer in layer_lst:
                                for lora_layers in lora_layer_lst:
                                    if task_name == "slurp":
                                        sfx_lst = ["scenario"]#, "action"]
                                    else:
                                        sfx_lst = [""]
                                    for sfx in sfx_lst:
                                        tot_runs += 1
                                        os.makedirs(task_name, exist_ok=True)
                                        generate_cmd(
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
                                            slurm_cmd,
                                            write_str_header,
                                            lora_layers,
                                            )
        if check_log:
            with open("output.csv", "w") as f:
                f.write("\n".join(log_write_str))
        else:
            write_cmd(f'exe_{task_name}_{adapter}_placement.sh', slurm_cmd, tot_runs)

run_sweep()



