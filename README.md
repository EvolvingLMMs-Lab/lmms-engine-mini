
# Synvo Engine

Training framework for Synvo.

## Current TODO
1. Support Flash-Attention for Qwen2AudioEncoder [Current Issue](https://github.com/QwenLM/Qwen2-Audio/issues/51)
2. [Long Term] Refactoring... (Possibly processing logic is the most needed one)

You can now run flash-attn and rmpad on Kino with Qwen2Audio. However, I did not fix the flash-attn forward in Qwen2Audio Attn but rather simple ignore it now. So if you enable the flash-attn, the flash-attn for audio encoder is actually `sdpa`

## Installation
Installation is simple
```bash
python3 -m pip install -e .
```

### Use rmpad
Rmpad is a techniques to accelerate the training process by removing the pad. With it enabled, it will boost the training performance quickly.

However, to use it, there are several restrictions:
1. You have to enable flash-attention
2. You have to build to rmsnorm for flash-attention from source

To use rmpad, you should install flash-attn also. You can do it by
```bash
python3 -m pip install flash-attn --no-build-isolation
```

Since we require the rmsnorm kernel, you need to build from source for flash-attn
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/csrc/layer_norm
python3 -m pip install .
# Or
# python3 -m pip install -v .
```

### Liger Kernel
[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput and reduces memory usage. Based on my testing, it does reduces memory usage when finetuning models. Benchmarking based on my testing under kino stage-1 training settings, it reduces the memory usage by around 30%.

To use it is simple, you need to first install it using `pip install liger-kernel`. Then set the `use_liger_kernel` in the trainer config to `true`. Make sure your model has a language model module and is in this [list](https://github.com/linkedin/Liger-Kernel/blob/61eefe9a4429459351979dc7fe1de746fd7ca86f/src/liger_kernel/transformers/monkey_patch.py#L795-L806) of modules

## Prepare Config
The overall design of our framework is that we build each component as a pipeline, you will need to pass in a config to use for init the pipeline.

An example config
```json
[
    {
        "type" : "trainer",
        "config" : {
            "trainer_type": "hf_trainer",
            "dataset_config": {
                "dataset_type" : "vision",
                "dataset_format" : "json",
                "dataset_path" : "./data/synvo_engine.json",
                "processor_config": {
                    "processor_name": "Qwen/Qwen2-VL-2B-Instruct",
                    "processor_modality": "vision",
                    "processor_type": "qwen2_vl"
                }
            },
            "model_config": {
                "model_name_or_path" : "Qwen/Qwen2-VL-2B-Instruct",
                "model_class" : "Qwen2VLForConditionalGeneration",
                "attn_implementation" : "flash_attention_2"
            },
            "per_device_train_batch_size": 1,
            "learning_rate": 5e-05,
            "weight_decay": 0.0,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 1,
            "save_steps": 1000,
            "report_to": "wandb",
            "output_dir": "./output",
            "warmup_ratio": 0,
            "run_name": "test_run",
            "logging_steps" : 1,
            "group_by_length" : true,
            "dataloader_num_workers" : 8,
            "bf16" : true
        }
    }
]
```

## Launch

Launching was being done by using `accelerate`.

```bash
# FSDP
CUDA_LAUNCH_BLOCKING=1 ACCELERATE_CPU_AFFINITY=1 accelerate launch \
    --use_fsdp \
    --mixed_precision bf16 \
    --fsdp_sharding_strategy HYBRID_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_forward_prefetch false \
    --fsdp_cpu_ram_efficient_loading true \
    --fsdp_offload_params false \
    --fsdp_state_dict_type SHARDED_STATE_DICT \
    --fsdp_sync_module_states true \
    --fsdp_transformer_layer_cls_to_wrap "SiglipVisionModel,Qwen2DecoderLayer" \
    --fsdp_use_orig_params true \
    --num_processes="8" \
    --num_machines="1" \
    --main_process_ip=<port_ip> \
    --main_process_port=<port> \
    --machine_rank="0" \
    -m synvo_engine.launch.cli --config scripts/config_custom.json
```

To launch it using deepspeed, you can

```bash
CUDA_LAUNCH_BLOCKING=1 ACCELERATE_CPU_AFFINITY=1 accelerate launch \
    --use_deepspeed \
    --mixed_precision bf16 \
    --deepspeed_config_file zero3.json \
    --num_processes="8" \
    --num_machines="1" \
    --main_process_ip=<port_ip> \
    --main_process_port=<port> \
    --machine_rank="0" \
    -m synvo_engine.launch.cli --config ${CONFIG}
```

You can also run deepspeed using `torchrun` and it is the recommended way when launch on multiple machines
```bash
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="<port_ip>" \
    --master_port="<port>" \
    -m synvo_engine.launch.cli --config ${CONFIG}
```

