# MiMo-V2-Flash

## Introduction

XiaomiMiMo/MiMo-V2-Flash, with 309B total parameters and 15B activated parameters, is a new inference-centric model designed to maximize decoding efficiency created by XiaomiMiMo Team explicitly co-designed for real-world serving workloads, enabling flexible tradeoffs between throughput and latency on different hardware.

This model creates a new balance between long-context modeling capability and inference efficiency. Key features include:
- **Hybrid Attention Architecture**: Interleaves Sliding Window Attention (SWA) and Global Attention (GA) with a 5:1 ratio and an aggressive 128-token window. This reduces KV-cache storage by nearly 6x while maintaining long-context performance via learnable attention sink bias.
- **Multi-Token Prediction (MTP)**: Equipped with a lightweight MTP module (0.33B params/block) using dense FFNs. This triples output speed during inference and will be good to accelerates rollout in RL training.
- **Efficient Pre-Training**: Trained on 27T tokens using FP8 mixed precision and native 32k seq length. The context window supports up to 256k length.
- **Agentic Capabilities**: Post-training utilizes Multi-Teacher On-Policy Distillation (MOPD) and large-scale agentic RL, achieving superior performance on SWE-Bench and complex reasoning tasks.


## Installation

MiMo-V2-Flash is currently available in SGLang via Docker image and pip install.

### Docker

```bash
# Pull the docker image
docker pull lmsysorg/sglang:dev-pr-15207

# Launch the container
docker run -it --gpus all \
  --shm-size=32g \
  --ipc=host \
  --network=host \
  lmsysorg/sglang:dev-pr-15207 bash
```

### Pip Installation

```bash
# On a machine with SGLang dependencies installed or inside a SGLang nightly container
# Start an SGLang nightly container
docker run -it --gpus all \
  --shm-size=32g \
  --ipc=host \
  --network=host \
  lmsysorg/sglang:nightly-dev-20251215-4449c170 bash

# If you already have SGLang installed, uninstall the current SGLang version
pip uninstall sglang -y

# Install the PyPI Package
pip install sglang==0.5.6.post2.dev8005+pr.15207.g39d5bd57a \
  --extra-index-url https://sgl-project.github.io/whl/pr/
```

## Model Deployment

Use the configuration selector below to automatically generate the appropriate deployment command.

import MiMoConfigGenerator from '@site/src/components/autoregressive/MiMoConfigGenerator';

<MiMoConfigGenerator />

MI355X (ROCm) is validated in the selector above with `--tp-size 4`, Triton attention, and `--disable-custom-all-reduce`. `--tp-size 8` hit a QKV sharding error during validation. EAGLE speculative decoding is still WIP on MI355X.

### AMD MI300X Docker Deployment

For AMD MI300X GPUs, use the official SGLang ROCm Docker image:

```bash
docker run -d --name sglang_mimo_v2_flash \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --ipc=host --shm-size 64g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 30000:30000 \
  -e SGLANG_USE_AITER=0 \
  -e USE_ROCM_AITER_ROPE_BACKEND=0 \
  lmsysorg/sglang:v0.5.11-rocm720-mi30x \
  bash -c "SGLANG_USE_AITER=0 USE_ROCM_AITER_ROPE_BACKEND=0 \
    python3 -m sglang.launch_server \
    --model XiaomiMiMo/MiMo-V2-Flash \
    --tp 2 --trust-remote-code \
    --mem-fraction-static 0.80 \
    --attention-backend triton \
    --disable-cuda-graph \
    --disable-custom-all-reduce \
    --host 0.0.0.0 --port 30000"
```

:::tip AMD MI300X Notes
- **TP=2**: MiMo-V2-Flash (309B total / 15B active MoE) requires at least 2 MI300X GPUs.
- **`--disable-cuda-graph`**: Required on MI300X to avoid FP8 Triton kernel compilation errors.
- **Environment Variables**: `SGLANG_USE_AITER=0` and `USE_ROCM_AITER_ROPE_BACKEND=0` are required for MLA model compatibility on AMD.
- **EAGLE speculative decoding**: Not yet supported on MI300X.
:::

## Testing the deployment

Once the server is running, test it with a chat completion request in another terminal:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "XiaomiMiMo/MiMo-V2-Flash",
    "messages": [
      {"role": "user", "content": "Hello! What can you help me with?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**Expected response:**

```json
{
  "id": "...",
  "object": "chat.completion",
  "model": "XiaomiMiMo/MiMo-V2-Flash",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hello! I can help you with..."
    }
  }]
}
```

## Troubleshooting

**DeepGEMM Timeout Error**

Occasionally DeepGEMM timeout errors occur during first launch. Simply rerun the server command in the same container - the compiled kernels are cached and subsequent launches will be fast.

**ROCm MI355X Attention Backend**

If you see an error such as `AiterAttnBackend.forward_decode() got an unexpected keyword argument 'sinks'` on MI355X, use the `MI355X` + `Performance Optimizations` command from the selector above, which switches to Triton attention and keeps `--disable-custom-all-reduce`.
