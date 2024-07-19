# CrashLLM

## Set up environment
We conduct experiments under CUDA 12.1 and Ubuntu 22.04 on Nvidia A100 GPU.

Create conda environment
```
conda create -n CLLM python=3.10
source activate CLLM
```
Install the related packages.
```
pip install -r requirements.txt
```

## Train

### Set the master_port and gpu devices
```
export master_port=50001
export include=localhost:0
```

### Set the model size
The model size will be set to 7b if `model_size` is not in the environment variables. 

```
export model_size=7b #7b, 13b, 70b
```

### Run the task-specific script
#### number_of_injuried_people: 

```
sh train/sft/train_scripts/inj.sh
```
#### severity: 

```
sh train/sft/train_scripts/sev.sh
```
#### accident_type: 

```
sh train/sft/train_scripts/type.sh
```

### Saved checkpoints

The best checkpoints (selected by the best eval_accuracy) and the last checkpoints will be saved at the end of the training stage. 

## Test

### Set the master_port and gpu devices
```
export master_port=50001
export include=localhost:0
```

### Set the model size
The model size will be set to 7b if `model_size` is not in the environment variables. 

```
export model_size=7b #7b, 13b, 70b
```

### Set the path of checkpoint.
```
export checkpoint_path=../../logs/7b_inj/checkpoint-900
```

### Run the task-specific script
#### number_of_injuried_people: 

```
sh train/sft/test_scripts/inj.sh
```
#### severity: 

```
sh train/sft/test_scripts/sev.sh
```
#### accident_type: 

```
sh train/sft/test_scripts/type.sh
```


## Acknowledgment

This repo is based on [Llama-Chinese](https://github.com/LlamaFamily/Llama-Chinese). 

