# VisTA

**VisualToolAgent (VisTA): A Reinforcement Learning Framework for Visual Tool Selection**

---

## 🎯 Overview

VisTA is a reinforcement learning framework designed to enhance visual tool selection capabilities in multimodal AI systems. Our approach focuses on training agents to intelligently select and utilize appropriate visual tools for complex reasoning tasks.

---

## 📋 Environment Setup

### Installation

```bash
conda create -n vista python=3.11
conda activate vista

pip install -e ".[dev]"
pip install wandb==0.18.3
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation
pip install vllm==0.7.2

pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
```

---

## 📊 Dataset Support

* **Currently Supported**

  * [ChartQA](https://github.com/vis-nlp/ChartQA)
* **Upcoming Support**

  * [Geometry3K](https://github.com/lupantech/InterGPS/tree/main)
* **Additional visual reasoning datasets with unified tooling interface** (coming soon)

---

## 🔧 Training

### Tool Selection Model Training

To train the visual tool selection model on ChartQA:

```bash
cd src/r1-v
./run_grpo.sh
```

---

## 🧪 Inference and Evaluation

### Generate Tool Predictions

Update the model path in `model_name_or_path` inside `run_grpo_test.sh`, then execute:

```bash
./run_grpo_test.sh
```

### Evaluate Tool-Based Reasoning

Run the following commands to evaluate:

```bash
python test_chartqa_gpt.py
```

```bash
python relax_test.py
```

---

## 📚 Citation

If you use **VisTA** in your research, please cite:

```bibtex
@misc{huang2025visualtoolagentvistareinforcementlearning,
  title={VisualToolAgent (VisTA): A Reinforcement Learning Framework for Visual Tool Selection}, 
  author={Zeyi Huang and Yuyang Ji and Anirudh Sundara Rajan and Zefan Cai and Wen Xiao and Junjie Hu and Yong Jae Lee},
  year={2025},
  eprint={2505.20289},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.20289},
}
```
