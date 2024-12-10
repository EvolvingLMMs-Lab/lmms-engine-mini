
# Synvo Engine

Training framework for Synvo.

## Current TODO
1. Support Audio
2. Test models ft such as Qwen-VL (from transformers or rmpad)
3. [Long Term] Refactoring... (Possibly processing logic is the most needed one)

## Installation
Installation is simple
```bash
python3 -m pip install -e .
```

### Use rmpad
To use rmpad, you should install flash-attn also. You can do it by
```bash
python3 -m pip install flash-attn --no-build-isolation
```

Since we require the rmsnorm kernel, you need to build from source for flash-attn
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd csrc/layer_norm
python3 -m pip install .
# Or
# python3 -m pip install -v .
```

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

