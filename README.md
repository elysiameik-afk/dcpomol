
# DCPO: Dynamic Clipping Policy Optimization
**GitHub repository:** https://github.com/Yshihui/DCPO 
---  

## Overview  
DCPO is a reinforcement‑learning‑from‑verifiable‑rewards (RLVR) framework that dramatically improves data‑utilisation and training speed for large language models (LLMs) on reasoning‑heavy tasks.  
All experiments from the paper *“DCPO: Dynamic‑Clipping Policy Optimization for Efficient Reinforcement Learning”* can be reproduced with this repo.

---  

## Key Contributions  

| Feature | What it does |
|---|---|
| **Dynamic adaptive clipping** | Computes a closed‑form clipping bound that depends on the old probability, reducing the token‑clipping ratio by ~10× compared with fixed  clipping. |
| **smooth advantage standardization** |Standardizes rewards by mixing the current‑step statistics with the cumulative statistics, removing zero‑gradient “dead zones” and increasing non‑zero‑gradient usage by ≈ 28 %. |
| **OTM loss** | Calculates the loss over tokens of a **single** response without batch-level averaging, preserving the relative advantage between responses. |
| **Broad‑scale validation** | Tested on MATH‑500, AMC‑23, AIME‑24, AIME‑25 with model sizes 1.5 B – 14 B. DCPO‑7B reaches **38.8 Avg@32** on AIME‑24 (↑ 21 % over GRPO) while halving wall‑clock GPU hours versus DAPO. |

---  

## Code base & Docker  

* **Code base** – DCPO extends the open‑source **Verl** codebase (https://github.com/volcengine/verl).  
  *mainly the loss formulation and the dynamic adaptive clipping / step‑smooth advantage standardization modules are added.*  

* **Docker image** – The training environment used in the paper is published as a Docker [image](https://hub.docker.com/layers/verlai/verl/app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2-deepep/images/sha256-172b68c83065c31d65d51855e45b580e5ea5998a5f0d7802023c31eb9e6243ad):  

```
docker pull verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2
```

## the stype of sample in  data parquet
```
[
    {
        "data_source": "qwen_aime_2024", # start with qwen
        "prompt":
        [
            {
                "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                "role": "system"
            },
            {
                "content": "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
                "role": "user"
            }
        ],
        "ability": "MATH",
        "reward_model":
        {
            "ground_truth": "25", # 
            "style": "rule-lighteval/MATH_v2"
        },
        "extra_info":
        {
            "index": 24,
            "raw_problem": "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
            "split": null
        }
    }
]
```

---  

## Quick Start  

### 1️⃣ Clone the repository  

```bash
git clone https://github.com/your-org/DCPO.git
cd DCPO
pip install -r requirements.txt
```

`requirements.txt` includes the additonal Python packages.

### 2️⃣ Download a pretrained checkpoint & data

```bash
# checkpoint (e.g. Qwen2.5‑Math‑7B) will be fetched from HuggingFace automatically
bash ./recipe/dcpo/run_dcpo.sh   # DCPO
bash ./recipe/dcpo/run_grpo.sh   # GRPO baseline
bash ./recipe/dcpo/run_dapo.sh   # DAPO baseline
bash ./recipe/dcpo/run_gspo.sh   # gspo albation baseline
```

Each script sets the corresponding hyperparameters and starts training (default is 8 GPUs for a single machine, and it automatically recognizes multiple machines)


---
---  

## 📊 Experimental Results  

- Avg@1: This metric represents the standard accuracy achieved using greedy decoding. It measures the performance of the model's single best prediction.
- Avg@32:  This metric calculates the average accuracy over 32 sampled responses per problem, using a temperature of 1.0 and top\_p of 1.0. This metric provides insight into the robustness and stability of the trained policy distribution.



### 1. Qwen‑Math‑1.5B-Instruct

| Model | MATH‑500 | AMC‑23 | AIME‑24 | AIME‑25 | **Average** |
|---|---|---|---|---|---|
| **Qwen‑Math‑1.5B-Instruct** | 73.6 | 57.5/49.4 | 10.0/10.0 | 3.3/6.1 | 36.1/21.8 |
| **GRPO** | **77.2** | 70.0/68.4 | 16.7/14.0 | 20.0/**13.5** | 46.0/32.0 |
| **DAPO** | 76.0 | **80.0**/70.6 | **20.0**/13.5 | 14.4/12.5 | 46.5/32.4 |
| **DCPO** | **77.2** | 75.0/**70.8** | **20.0**/**15.6** | 16.7/12.1 | **47.2**/**32.8** |
| **Δ GRPO** | +0.0 | +5.0/+2.4 | +3.3/+1.6 | ‑3.3/‑1.4 | +1.2/+0.8 |
| **Δ DAPO** | +1.2 | ‑5.0/+0.2 | +0.0/+2.1 | +0.0/+2.1 | +0.7/+0.4 |

### 2. Qwen‑3B

| Model | MATH‑500 | AMC‑23 | AIME‑24 | AIME‑25 | **Average** |
|---|---|---|---|---|---|
| **Qwen‑3B (Common base)** | 46.4 | 27.5/7.8 | 3.3/0.1 | 3.3/0.7 | 20.1/2.9 |
| **GRPO** | 69.2 | 62.5/51.6 | **10.0**/7.5 | 6.7/4.5 | 36.3/21.0 |
| **DAPO** | 72.4 | 57.5/54.0 | **10.0**/**8.3** | 6.7/3.9 | 36.6/**23.1** |
| **DCPO** | **71.2**| **62.5**/55.8 | 3.3/7.5 | **10.0**/**4.7** | **37.6**/22.7 |
| **Δ GRPO** | +2.0 | **+0.0**/**+4.2** | ‑6.7/**+0.0** | **+3.3**/**+0.2** | **+1.3**/**+1.7** |
| **Δ DAPO** | ‑1.2 | **+5.0**/**+1.8** | ‑6.7/‑0.8 | **+3.3**/**+0.8** | **+1.0**/‑0.4 |

### 3. Qwen‑Math‑7B & Qwen‑14B

| Model | MATH‑500 | AMC‑23 | AIME‑24 | AIME‑25 | **Average** |
|---|---|---|---|---|---|
| **Qwen‑Math‑7B (Math base)** | 50.4 | 40.0/19.5 | 13.3/6.0 | 3.3/1.5 | 28.4/9.3 |
| **GRPO** | 81.6 | 77.5/75.9 | 36.7/32.1 | 16.7/16.7 | 53.1/41.6 |
| **DAPO** | **83.0** | 72.5/**80.7** | 36.7/31.6 | **23.3**/14.9 | 53.9/42.4 |
| **DCPO** | 82.5 | **82.6**/79.8 | **46.7**/**38.8** | 16.7/**17.2** | **57.1**/**45.2** |
| **Δ GRPO** | +0.9 | +5.1/+4.9 | +10.0/+6.7 | **+0.0**/+0.5 | +4.0/+3.6 |
| **Δ DAPO** | ‑0.5 | **+10.1**/‑0.9 | **+10.0**/**+7.2** | **‑6.6**/**+2.3** | **+3.3**/**+2.8** |

| Model | MATH‑500 | AMC‑23 | AIME‑24 | AIME‑25 | **Average** |
|---|---|---|---|---|---|
| **Qwen‑Math‑14B (Common base)** | 60.8 | 47.5/16.4 | 3.3/1.3 | 3.3/1.1 | 28.7/6.3 |
| **GRPO** | 81.2 | 75.0/65.6 | 13.3/17.6 | 13.3/10.5 | 45.7/31.3 |
| **DAPO** | 83.4 | **87.5**/**85.1** | 16.7/16.4 | 20.0/15.3 | 51.9/38.9 |
| **DCPO** | **84.6** | 85.0/79.9 | **20.0**/**18.2** | **23.3**/**19.0** | **53.2**/**39.0** |
| **Δ GRPO** | **+3.4** | **+10.0**/**+14.3** | **+6.7**/**+0.6** | **+10.0**/**+8.5** | **+6.5**/**+7.7** |
| **Δ DAPO** | **+1.2** | ‑2.5/‑5.2 | **+3.3**/**+1.8** | **+3.3**/**+3.7** | **+1.3**/**+0.1** |

#### Take‑away  

* **Response Utilisation Ratio (RUR)** ↑ ≈ 70 % for DCPO (vs. ≈ 44 % for GRPO).  
* **Token‑Clipping Ratio (TCR)** is reduced 10× lower than GRPO/DAPO.  
* **Training wall‑clock time** is roughly half of DAPO for the same number of update steps.  

---  
## License

The source code is released under the **Apache‑2.0 license** (the same license as the underlying Verl code).
Pre‑trained Qwen‑Math checkpoints or others are provided under their original licenses – please refer to the model cards on HuggingFace for details.

---

## Citation

If you use DCPO in your research, please cite the original work:

```bibtex
@inproceedings{yourname2025dcpo,
  title   = {DCPO: Dynamic‑Clipping Policy Optimization for Efficient Reinforcement Learning},
  author  = {Shihui Yang, },
  organization = {Baichuan inc.},
  year    = {2025},
  url     = {https://github.com/your-org/DCPO},
  note    = {Open‑source implementation},
}
```

---

**Happy hacking!**
If you run into any issues, open a GitHub *Issue* or start a discussion in the repository.
