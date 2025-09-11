# DCPO: Dynamic Clipping Policy Optimization

📝 [Paper@arXiv](https://arxiv.org/abs/2509.02333) | 🤗 [HuggingFace](https://huggingface.co/papers/2509.02333) | 🐱 [GitHub](https://github.com/lime-RL/DCPO)

---

## 1.Paper of this work.

All experiments from the paper *“DCPO: Dynamic Clipping Policy Optimization”* can be reproduced with this repo.

DCPO is a reinforcement‑learning‑from‑verifiable‑rewards (RLVR) framework that dramatically improves data‑Utilization and training speed for large language models (LLMs) on reasoning‑heavy tasks.

---

## 2. Key Contributions

|                  Feature                  | What it does                                                                                                                                                                                                   |
| :----------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    **Dynamic Adaptive Clipping**    | Computes a closed‑form clipping bound that depends on the old probability, reducing the token‑clipping ratio by ~10× compared with fixed  clipping.                                                         |
| **Smooth Advantage Standardization** | Standardizes rewards by mixing the current‑step statistics with the cumulative statistics, removing zero‑gradient “dead zones” and increasing non‑zero‑gradient usage by ≈ 28 %.                     |
|             **OTM loss**             | Calculates the loss over tokens of a**single** response without batch-level averaging, preserving the relative advantage between responses.                                                              |
|     **Broad‑scale validation**     | Tested on MATH‑500, AMC‑23, AIME‑24, AIME‑25 with model sizes 1.5 B – 14 B. DCPO‑7B reaches**38.8 Avg@32** on AIME‑24 (↑ 21 % over GRPO) while halving wall‑clock GPU hours versus DAPO. |

---

## 3. Preliminary

### GRPO

$$
\mathcal{T}_{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left( r_{i,t}\left(\theta\right)\hat{A}_{i,t}, \text{clip}(r_{i,t}\left(\theta\right),1-\epsilon,1+\epsilon)\hat{A}_{i,t} \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

GRPO first samples G responses for each query, assigns rewards R through a rule-based reward function and estimates token-level advantage.

### DAPO

$$
\begin{aligned}
    \mathcal{T}_{\mathrm{DAPO}}\left(\theta\right)&={\frac{1}{\sum_{i=1}^G|o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|}}\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\mathrm{~clip}\left(r_{i,t}(\theta),1-{\epsilon_{\mathrm{low}}},1+{\epsilon_{\mathrm{high}}}\right)\hat{A}_{i,t}\right)\\
& where,\ 0 <\left|\{o_i\mid{is\_equivalent}(a,o_i)\}\right|<G
\end{aligned}
$$

$0 <\left|\{o_i\mid{is\_equivalent}(a,o_i)\}\right|<G$ means DAPO will discard the all responses while same reward and regenerates responses to maintain batch size.

### GSPO

$$
\begin{aligned}
    \mathcal{T}_{\mathrm{GSPO}}\left(\theta\right)=&\frac{1}{G}\sum_{i=1}^G\min\left( s_{i}\left(\theta\right)\hat{A}_{i}, \text{clip}(s_{i}\left(\theta\right),1-\epsilon,1+\epsilon)\hat{A}_{i} \right) \\
    &\text{where\ } s_i(\theta) = % \left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}\right)^{\frac{1}{|o_i|}}=
    exp(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}log(\frac{\pi_\theta(o_{j,t}|q)}{\pi_{\theta_{old}}(o_{j,t}|q)})) 
\end{aligned}
$$

**GSPO replace the token-level clipping methods with sequence-level clipping, and then discards the nonzero-advantage responses with high-variance, while it  will keep the tokens with high token-level probability ratio resulting in the training instability, and waste much informative token in high-variance responses.**

### Prior Advantage Calculation

$$
\hat{A}_{j,t}^i = \frac{\left(R^i_j-\mu^i\right)}{\sigma^i} \\
$$

In previous works, including GRPO and DAPO, the advantage $\hat{A}_{j,t}^i$ for the token $t$ in the response $j$ is calculated by standardizing the reward $R^i_j$  against the mean $\mu^i$ and standard deviation $\sigma^i$ of the rewards of the $G$ responses generated in the $i$-th step. When the rewards for the same prompt are identical, responses with zero advantages do not contribute to model update, resulting in response waste.

## 4.Details of major innovations

### 4.1 Dynamic Adaptive Clipping（DAC)

For Importance sampling, the expected value of a function $f(x)$ under the new probability $p(x)$ can be rewritten as an expectation under the old probability $q(x)$ by importance sampling weight. Although this estimator is unbiased, its variance can be significantly inflated, which is a common challenge in importance sampling.

$$
\begin{aligned}
    {Var}_{x \sim q}\left[f(x)\frac{p(x)}{q(x)}\right]- {Var}_{x \sim p}\left[f(x)\right] =&\mathbb{E}_{x \sim p}\left[f(x)^2(\frac{p(x)}{q(x)}-1)\right] =\int f(x)^2(\frac{p(x)}{q(x)}-1)p(x)\mathrm{d}x
    \end{aligned}
$$

Previous works(e.g., PPO, GRPO)usually set the fixed bounds $\epsilon$ for the $|\frac{p(x)}{q(x)}-1|$ to limit the variance bias.  this method don't consider the different probabbility across different tokens,resulting in a smaller effective absolute space with smaller probabilities. This is unreasonable because the less confident the model is about a token (with the lower probability), the more valuable information it can provide to model. we propose a more practical alternative to constrain the probability ratio $r(x)$ through the dynamic-adaptive mechanism by including the probability in the restriction as $|(\frac{p(x)}{q(x)}-1)p(x)|\le \epsilon $. Finally, we get the dynamic-adaptive clipping bounds which adaptively adjust the bounds of $r(x)$ with different old probability.

$$
\begin{aligned}
        0.5+\frac{1}{2}\sqrt{\max\left(1-\frac{4\epsilon_{low}}{q\left(x\right)},\ 0\right)}\leq&r\left(x\right) \leq 0.5+\frac{1}{2}\sqrt{1+\frac{4\epsilon_{high}}{q\left(x\right)}}
    \end{aligned}
$$

<div style="display:flex; justify-content:center; gap:2%; flex-wrap:wrap;">
  <img src="https://arxiv.org/html/2509.02333v2/x5.png" alt="图 1" style="width:48%;">
  <img src="https://arxiv.org/html/2509.02333v2/x6.png" alt="图 2" style="width:48%;">
</div>
<div style="display:flex; justify-content:center; gap:2%; flex-wrap:wrap;">
  <img src="https://arxiv.org/html/2509.02333v2/x7.png" alt="图 1" style="width:48%;">
  <img src="https://arxiv.org/html/2509.02333v2/x8.png" alt="图 2" style="width:48%;">
</div>


|  different clipping method  |              clip thresholds              | q(x) |    low p(x)    |   high q(x)   |
| :--------------------------: | :-----------------------------------------: | :--: | :-------------: | :------------: |
| symmetric fixed bound(GRPO) |              $\epsilon=0.2$              | 0.9 |      0.72      |  min(1.08,1)  |
| asymmetric fixed bound(DAPO) | $\epsilon_{low}=0.2,\epsilon_{high}=0.28$ | 0.9 |      0.72      |  min(1.152,1)  |
| dynamic-adaptive bound(Our) | $\epsilon_{low}=0.16,\epsilon_{high}=0.2$ | 0.9 | **0.69** |  min(1.06,1)  |
| symmetric fixed bound(GRPO) |              $\epsilon=0.2$              | 0.01 |      0.008      |     0.0012     |
| asymmetric fixed bound(DAPO) | $\epsilon_{low}=0.2,\epsilon_{high}=0.28$ | 0.01 |      0.008      |    0.00128    |
| dynamic-adaptive bound(Our) | $\epsilon_{low}=0.16,\epsilon_{high}=0.2$ | 0.01 | **0.005** | **0.05** |

As we can see, when the probability is small, the high q(x)  of our dynamic-adaptive clipping method could participate in model update is much greater than the fixed methods (whether asymmetric or symmetric fixed methods).

### 4.2 Smooth Advantage Standardization(SAS)

Previous Works calculate the advantage only considering the current-step rewards of generated responses. This approach can lead to several issues:

1. when randomness in response sampling causes all rewards to be the same at a given step, the advantage becomes zero, preventing the prompt from contributing to parameter updates despite potentially valuable differences in reasoning trajectories.
2. randomness in high-entropy sampling can yield highly skewed label counts, causing large fluctuations in standardized advantage values across steps, even reversing signs, thus destabilizing training.

We consider the cumulative rewards for the same prompt to calculate the advantage.

$$
\begin{aligned}
        \hat{A}_{total,j}^i=\frac{\left(R^i_j-\mu_{total}^i\right)}{\sigma_{total}^i} \\
    \end{aligned}
$$

To mitigate fluctuations of the step-specific standardization ${\hat{A}^i_{new,j}}$ and the cumulative standardization ${\hat{A}^i_{total,j}}$, we introduce two smoothing functions, ${\hat{SA}^i_{new,j}}$ and ${\hat{SA}^i_{total,j}}$, which represent the weighted average between the two standardization methods with the weights changing over step $i$

$$
\hat{SA}^i_{new,j} = \frac{i-1}{i}\hat{A}_{new,j}^i + \frac{1}{i}\hat{A}_{total,j}^i,\ \hat{SA}^i_{total,j} = \frac{1}{i}\hat{A}_{new,j}^i + \frac{i-1}{i}\hat{A}_{total,j}^i
$$

To reduce the impact of the respective fluctuations of cumulative standardization and standardization of the current step on training stability, our final advantage $\hat{A}^i_j$ is defined as the smoothed advantage with the smaller absolute value.

$$
\hat{A}^i_j=\begin{cases} \hat{SA}^i_{new,j} , & \text{when} \ |\hat{SA}^i_{new,j}| < |\hat{SA}^i_{total,j}|\\
        \hat{SA}^i_{total,j} , & \text{otherwise}
    \end{cases}
$$

Once the prompt participates in model optimization, the response of this prompt will  participate in model update in the later steps. When the rewards are same in current steps, they will participate with advantage $\frac{1}{i}\hat{A}_{total,j}^i$.


## 5. Code base & Docker

* **Code base** – DCPO extends the open‑source **Verl** codebase (https://github.com/volcengine/verl).*mainly the loss formulation and the dynamic adaptive clipping / step‑smooth advantage standardization modules are added.*
* **Docker image** – The training environment used in the paper is published as a Docker [image](https://hub.docker.com/layers/verlai/verl/app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2-deepep/images/sha256-172b68c83065c31d65d51855e45b580e5ea5998a5f0d7802023c31eb9e6243ad):

```
docker pull verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2
```

## 6. the example of sample in  data parquet

```
[
    {
        "data_source": "qwen_aime_2024", # start with qwen
        "prompt":
        [
            {
                "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                "role": "system"
            }, # the qwen template
            {
                "content": "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.", # question
                "role": "user"
            }
        ],
        "ability": "MATH",
        "reward_model":
        {
            "ground_truth": "25", # the label for the question 
            "style": "rule-lighteval/MATH_v2" # option
        },
        "extra_info":
        {
            "index": 24, # must have and be different from each other 
            "raw_problem": "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
            "split": null
        }
    }
]
```

---

## 7. Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/lime-RL/DCPO.git
cd DCPO
pip install -r requirements.txt
```

`requirements.txt` includes the additional Python packages.

### 2️⃣ Download a pre-trained checkpoint & data

```bash
# checkpoint (e.g. Qwen2.5‑Math‑7B) will be fetched from HuggingFace automatically
# change your data to the template and modify the path on *.sh
bash ./recipe/dcpo/run_dcpo.sh   # DCPO
bash ./recipe/dcpo/run_grpo.sh   # GRPO baseline
bash ./recipe/dcpo/run_dapo.sh   # DAPO baseline
bash ./recipe/dcpo/run_gspo.sh   # gspo albation baseline
```

Each script sets the corresponding hyper-parameters and starts training (default is 8 GPUs for a single machine, and it automatically recognizes multiple machines)

---

---

## 8. Experimental Results

- Avg@1: This metric represents the standard accuracy achieved using greedy decoding. It measures the performance of the model's single best prediction.
- Avg@32:  This metric calculates the average accuracy over 32 sampled responses per problem, using a temperature of 1.0 and top\_p of 1.0. This metric provides insight into the robustness and stability of the trained policy distribution.

### 1. Qwen2.5‑Math‑1.5B-Instruct

|       Model       | MATH‑500      | AMC‑23             | AIME‑24                      | AIME‑25            | **Average**             |
| :----------------: | -------------- | ------------------- | ----------------------------- | ------------------- | ----------------------------- |
|        base        | 73.6           | 57.5/49.4           | 10.0/10.0                     | 3.3/6.1             | 36.1/21.8                     |
|   **GRPO**   | **77.2** | 70.0/68.4           | 16.7/14.0                     | 20.0/**13.5** | 46.0/32.0                     |
|   **DAPO**   | 76.0           | **80.0**/70.6 | **20.0**/13.5           | 14.4/12.5           | 46.5/32.4                     |
|   **DCPO**   | **77.2** | 75.0/**70.8** | **20.0**/**15.6** | 16.7/12.1           | **47.2**/**32.8** |
| **Δ GRPO** | +0.0           | +5.0/+2.4           | +3.3/+1.6                     | ‑3.3/‑1.4         | +1.2/+0.8                     |
| **Δ DAPO** | +1.2           | ‑5.0/+0.2          | +0.0/+2.1                     | +0.0/+2.1           | +0.7/+0.4                     |

### 2. Qwen2.5‑3B (Common base)

| Model              | MATH‑500      | AMC‑23                       | AIME‑24                     | AIME‑25                      | **Average**             |
| ------------------ | -------------- | ----------------------------- | ---------------------------- | ----------------------------- | ----------------------------- |
| base               | 46.4           | 27.5/7.8                      | 3.3/0.1                      | 3.3/0.7                       | 20.1/2.9                      |
| **GRPO**     | 69.2           | 62.5/51.6                     | **10.0**/7.5           | 6.7/4.5                       | 36.3/21.0                     |
| **DAPO**     | 72.4           | 57.5/54.0                     | **10.0**/**8.3** | 6.7/3.9                       | 36.6/**23.1**           |
| **DCPO**     | **71.2** | **62.5**/55.8           | 3.3/7.5                      | **10.0**/**4.7**  | **37.6**/22.7           |
| **Δ GRPO** | +2.0           | **+0.0**/**+4.2** | ‑6.7/**+0.0**         | **+3.3**/**+0.2** | **+1.3**/**+1.7** |
| **Δ DAPO** | ‑1.2          | **+5.0**/**+1.8** | ‑6.7/‑0.8                  | **+3.3**/**+0.8** | **+1.0**/‑0.4          |

### 3. Qwen2.5‑Math‑7B (Math base)

| Model              | MATH‑500      | AMC‑23                       | AIME‑24                       | AIME‑25                       | **Average**             |
| ------------------ | -------------- | ----------------------------- | ------------------------------ | ------------------------------ | ----------------------------- |
| base               | 50.4           | 40.0/19.5                     | 13.3/6.0                       | 3.3/1.5                        | 28.4/9.3                      |
| **GRPO**     | 81.6           | 77.5/75.9                     | 36.7/32.1                      | 16.7/16.7                      | 53.1/41.6                     |
| **DAPO**     | 83.0           | 72.5/**80.7**           | 36.7/31.6                      | **23.3**/14.9            | 53.9/42.4                     |
| **GSPO**     | **84.0** | 80.0/78.8                     | 40.0/34.9                      | 16.7/16.2                      | 55.2/43.3                     |
| **DCPO**     | 82.5           | **82.6**/79.8           | **46.7**/**38.8**  | 16.7/**17.2**            | **57.1**/**45.2** |
| **Δ GRPO** | +0.9           | +5.1/+4.9                     | +10.0/+6.7                     | **+0.0**/+0.5            | +4.0/+3.6                     |
| **Δ DAPO** | ‑0.5          | **+10.1**/‑0.9         | **+10.0**/**+7.2** | **‑6.6**/**+2.3** | **+3.3**/**+2.8** |
| **Δ GSPO** | ‑1.5          | **+2.6**/**+1.0** | **+6.7**/**+3.9**  | **+0.0**/**+1.0**  | **+1.9**/**+1.9** |

### 4. Qwen2.5‑14B (Common base)

| Model                                   | MATH‑500      | AMC‑23                         | AIME‑24                      | AIME‑25                       | **Average**             |
| --------------------------------------- | -------------- | ------------------------------- | ----------------------------- | ------------------------------ | ----------------------------- |
| **Qwen‑Math‑14B (Common base)** | 60.8           | 47.5/16.4                       | 3.3/1.3                       | 3.3/1.1                        | 28.7/6.3                      |
| **GRPO**                          | 81.2           | 75.0/65.6                       | 13.3/17.6                     | 13.3/10.5                      | 45.7/31.3                     |
| **DAPO**                          | 83.4           | **87.5**/**85.1**   | 16.7/16.4                     | 20.0/15.3                      | 51.9/38.9                     |
| **GSPO**                          | 78.6           | 77.5/75.0                       | **23.3**/16.0           | 16.7/9.9                       | 49.0/33.5                     |
| **DCPO**                          | **84.6** | 85.0/79.9                       | 20.0/**18.2**           | **23.3**/**19.0**  | **53.2**/**39.0** |
| **Δ GRPO**                      | **+3.4** | **+10.0**/**+14.3** | **+6.7**/**+0.6** | **+10.0**/**+8.5** | **+6.5**/**+7.7** |
| **Δ DAPO**                      | **+1.2** | ‑2.5/‑5.2                     | **+3.3**/**+1.8** | **+3.3**/**+3.7**  | **+1.3**/**+0.1** |
| **Δ GSPO**                      | **+6.0** | **+7.5**/**+4.9**   | -3.3/**+2.2**           | **+6.6**/**+10.1** | **+4.2**/**+5.5** |

#### Take‑away

* **Response Utilization Ratio (RUR)** ↑ 70 % for DCPO (vs.  44 % for GRPO).
* **Token‑Clipping Ratio (TCR)** is reduced 10× lower than GRPO/DAPO.
* **Training wall‑clock time** is roughly half of DAPO for the same number of update steps.

### 8.1. Token Clipping Ratio(TCR)

$$
\text{TCR} =Average\left(\sum_{m=1}^{N}\frac{\text{Number of clipped tokens in\ }micro_m}{\text{Total number of tokens in\ } micro_m}\right)
$$

<div style="display:flex; justify-content:center; gap:2%; flex-wrap:wrap;">
  <img src="https://arxiv.org/html/2509.02333v2/x2.png" alt="图 1" style="width:80%;">
</div>


- GRPO: for smaller model( 1.5B and 3B), the TCR increases with training steps. while for larger models(7B and 14B), it gradually decreases.
- DAPO: DAPO shows an upward trajectory for all model scales. it indicates that DCPO will have more proportion of partial or truncated responses to update models.
- GSPO: the TCR based on Qwen2.5-Math-7B is greater than 11% and based on Qwen2.5-14B is over 15%, which is much higher than the token-level clipping methods, resulting in most responses been wasted during training. Although GSPO keep sequence-level variance-bias is small, it keep some tokens with high token-level importance ratio, and these tokens may Increase training instability.
- DCPO: different with both GRPO and DAPO, the TCR for DCPO remains relatively constant and an order of magnitude lower than that of GRPO and DAPO. **DCPO use more high-entropy tokens which are more informative, while discarding some tokens with excessively abnormal importance weight. DCPO frees up more reasonable space for model exploration.**

> Taking Qwen2.5-Math-7B as an example, we observe that after about 60 steps, the probability of about 95% in generated tokens more than 0.9, and after about 100 steps, it more than 97%, and increases in later stages of training. this indicates that model generate most token with high confidence but a few token with high-entropy. so the most of token-level

### 8.2 Response Utilization Ratio(RUR)

<div style="display:flex; justify-content:center; gap:2%; flex-wrap:wrap;">
  <img src="https://arxiv.org/html/2509.02333v2/x3.png" alt="图 1" style="width:40%;">
</div>

| model                      | GRPO  | GSPO  | DCPO  |
| -------------------------- | ----- | ----- | ----- |
| Qwen2.5-Math-1.5B-Instruct | 45.6% | -     | 67.1% |
| Qwen2.5-3B                 | 48.3% | -     | 74.3% |
| Qwen2.5-Math-7B            | 37.4% | 43.5% | 73.2% |
| Qwen2.5-14B                | 43.9% | 47.6% | 72.4% |
| Average                    | 43.8% | 45.6% | 71.8% |


Due to average RUR of GRPO and GSPO is less than 50%,  GRPO and GSPO waste more than half of generated responses under the current-step standardization. but our method DSPO keep around 70% after first epoch, and keep slowly increasing in the subsequent training steps.

### 8.3 Ablation Result

<div style="display:flex; justify-content:center; gap:2%; flex-wrap:wrap;">
  <img src="https://arxiv.org/html/2509.02333v2/x4.png" alt="图 1" style="width:40%;">
</div>


We conducted an ablation study on Qwen2.5-Math-7B to assess the contribution of each component in DCPO, using Avg@32 as the evaluation metric. This metric highlights the robustness and stability of the learned policy distribution. To ensure fairness, each experiment modifies a single component of the baseline GRPO framework while keeping all other settings identical and removing the KL divergence term to align with DAPO, GSPO, and the full DCPO.

Each component of DCPO contributes positively to overall performance, and their combination leads to substantial cumulative gains. The results validate the effectiveness of the proposed mechanisms in improving data efficiency and stability in reinforcement learning for LLMs.

## 9. License

The source code is released under the **Apache‑2.0 license** (the same license as the underlying Verl code).
Pre‑trained Qwen‑Math checkpoints or others are provided under their original licenses – please refer to the model cards on HuggingFace for details.

---

## 10. Citation

If you use DCPO in your research, please cite the original work:

```bibtex
@misc{yang2025dcpodynamicclippingpolicy,
      title={DCPO: Dynamic Clipping Policy Optimization}, 
      author={Shihui Yang and Chengfeng Dou and Peidong Guo and Kai Lu and Qiang Ju and Fei Deng and Rihui Xin},
      year={2025},
      eprint={2509.02333},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.02333}, 
}
```

---

**Happy hacking!**
If you run into any issues, open a GitHub *Issue* or start a discussion in the repository.
