Environment Setup

# Create and activate the environment
conda create -n r1-v python=3.11
conda activate r1-v

# Navigate to the source directory
cd src/r1-v

# Install project dependencies
pip install -e ".[dev]"
pip install wandb==0.18.3
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation
pip install vllm==0.7.2

# Use specific transformers commit for compatibility
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef





Dataset Usage
Currently Supported: ChartQA
Additional datasets (e.g., Geometry3K) will be supported in the future.


Training
To train the tool selection model on ChartQA:
./run_grpo.sh

Generate tool predictions
Update the model path in model_name_or_path inside run_grpo_test.sh, then run:
./run_grpo_test.sh

Evaluate tool-based reasoning
Run GPT-based evaluation scripts:
python test_chartqa_gpt.py
python relax_test.py