"""
Generate slurm train scripts for adapter and LoRA experiments
"""

import os

import numpy as np

write_str = []
write_str.append("#! /bin/bash")
write_str.append("#SBATCH --partition=speech-gpu")
write_str.append("#SBATCH --gpus=1")
write_str.append("#SBATCH --cpus-per-task=6")
write_str.append("#SBATCH --array=0-1%1")
write_str.append("#SBATCH --constraint=24g")

write_str.append('eval \"$(/share/data/speech/ankitap/miniconda3/bin/conda \"shell.bash\" \"hook\")\"')
write_str.append("conda activate s3-adapt")

write_str_header = "\n".join(write_str)

def write_cmd(save_fn, slurm_cmd, tot_runs):
    with open(f'../{save_fn}', 'w') as f:
        f.write("\n".join(slurm_cmd))
    print(f"{len(slurm_cmd)} of {tot_runs} written to {save_fn}")

def check_status(task_name, fname):
    log_fn = os.path.join(f"../result/downstream/{task_name}-exp", fname, "log.log")
    if task_name == "slue":
        tot_steps = "20000"
    elif 'ctc' in task_name:
        tot_steps = "100000"
    else:
        tot_steps = "200000"
    if os.path.exists(log_fn):
        with open(log_fn, "r") as f:
            log_data = f.readlines()
        if tot_steps in log_data[-1]:
        # if f"test at step {tot_steps}" in log_data[-1]:
            return False
    return True

def generate_cmd(task_name, model_name, sfx, lr, layer_num, adapter, slurm_cmd):
    if adapter == 'lora':
        model_name += '_cp'
    if task_name == "slurp":
        write_fn = f"{model_name}-lr{lr}-{adapter}-{sfx}"
        cfg_str = f",,config.downstream_expert.datarc.classname={sfx}"
    else:
        write_fn = f"{model_name}-lr{lr}-{adapter}"
        cfg_str = ""
    if layer_num != 'all':
        write_fn += f'-layer{layer_num}'
    run_flag = check_status(task_name, write_fn)
    if run_flag:
        cmd_str = ""
        if adapter == 'houlsby':
            cmd_str += f"for i in $(seq 1 10);\n\tdo python3 run_downstream.py --adapter {adapter} -m train -d {task_name} -a -o "
        elif adapter == 'lora':
            cmd_str += f"for i in $(seq 1 10);\n\tdo python3 run_downstream_cp.py --adapter {adapter} -m train -d {task_name} -a -o "
        cmd_str += f"\"config.optimizer.lr={lr}{cfg_str}\" -u {model_name}"
        if layer_num != 'all':
            cmd_str += f' -l {layer_num}'
        if task_name == "ctc":
            cmd_str += f" -c downstream/ctc/libriphone.yaml"
        cmd_str += f" -n ./{task_name}-exp/{write_fn}"
        cmd_str += ";\ndone"
        # write_str.append(cmd_str)
        with open(os.path.join(task_name, f"{write_fn}.sh"), "w") as f:
            f.write("\n".join([write_str_header, cmd_str]))
        slurm_cmd.append(f"sbatch slurm-scripts/{task_name}/{write_fn}.sh")

def run_all(check_log=False):
    # model_lst = ["wav2vec2", "wav2vec2_large_ll60k", "hubert", "hubert_large_ll60k"]
    # task_lst = ["asr", "ctc", "slue_vp_ner"]
    task_lst = ["slurp"]
    lr_lst = [0.00005, 0.0001, 0.0002, 0.0005]
    # task_lr_tuple = [(task, lr) for task, lr in zip(task_lst, lr_lst)]
    task_lr_tuple = [(task, lr) for task in task_lst for lr in lr_lst]
    print(task_lr_tuple)
    import pdb; pdb.set_trace()
    model_lst = ["hubert"]
    adapter_lst = ["houlsby", "lora"]
    layer_lst = ['all']
    # layer_lst = [9, 10, 11, 12, 'all']
    slurm_cmd = []
    log_write_str = []
    tot_runs = 0
    for model_name in model_lst:
        for layer_num in layer_lst:
            for task_name, lr in task_lr_tuple:
                if task_name == "slurp":
                    sfx_lst = ["scenario"]#, "action"]
                else:
                    sfx_lst = [""]
                for sfx in sfx_lst:
                    for adapter in adapter_lst:
                        if check_log:
                            pass
                            # log_line = process_log_file(task_name, model_name, sfx, lr, layer_num)
                            # log_write_str.append(log_line)
                        else:
                            tot_runs += 1
                            os.makedirs(task_name, exist_ok=True)
                            generate_cmd(task_name, model_name, sfx, lr, layer_num, adapter, slurm_cmd)
    
    if check_log:
        with open("output.csv", "w") as f:
            f.write("\n".join(log_write_str))
    else:
        write_cmd('exe_slurp.sh', slurm_cmd, tot_runs)

run_all()