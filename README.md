# *Autonomous Circuit Design: Qwen Netlist-to-Placement Pretraining with GRPO*

**version 1.0**

**Date:2025/12/7**

**Author: Kailiang**

## Abstract

We propose the first end-to-end placement framework that integrates a large language model (LLM) adapted via parameter-efficient adapters with Group Relative Policy Optimization (GRPO) to automate FPGA design stages from natural-language specifications to Verilog, gate-level netlists, and physical placement. The system treats the LLM as a code-to-code translator and hint generator, while a GRPO-trained policy produces placement candidates conditioned purely on the textual netlist structure and LLM hints. A novel two-stage training protocol is adopted: first, a supervised pretraining stage on *Netlist→Placement* pairs using **Qwen** (textual netlist serialization and placement formats) to teach the LLM placement-conditioned outputs; second, a reinforcement learning stage where the base **Qwen** weights are frozen, and the LoRA/adapter parameters along with a lightweight policy head are updated with GRPO using the composite reward defined in this paper. To enable reproducibility we define a canonical netlist/placement JSON representation, describe GRPO adaptations (group normalization, hierarchical actions, surrogate evaluators), and present a multi-objective, normalized reward composed of timing, HPWL, congestion, overflow, power and legality terms. Implementation uses Yosys, VTR/VPR and OpenROAD. 

-----

![image](framework2.png)

## 1\. Introduction

Automating hardware design flows — from high-level specifications to physical placement and routing — remains a central challenge in electronic design automation (EDA). The traditional toolchain splits the problem into multiple stages (synthesis, netlist generation, packing, placement, routing), each relying on specialized heuristics and algorithmic optimizations. Recent progress in large language models (LLMs) demonstrates the ability of these models to perform structured code generation and translation tasks across formats, suggesting the opportunity to use LLMs as flexible, semantic-aware components within an EDA pipeline (see Hu et al., 2021; Qwen model release).

This work extends that opportunity in two directions:

1.  **Netlist→Placement supervised pretraining** for a text-capable LLM (**Qwen**) so it learns to emit region-level placement hints and coarse coordinate outputs directly from a canonical netlist serialization.
2.  **Pure Textual Representation Integration**: We rely exclusively on the LLM's ability to process long-context textual serializations of netlists, removing the need for auxiliary graph encoders. The transformer input consists solely of text tokens representing the design constraints and netlist topology.

To preserve the pretrained capabilities of **Qwen** we freeze base weights in RL fine-tuning, train parameter-efficient adapters (LoRA) and a small policy head, and optimize with GRPO under the composite reward defined below.

**Contributions (summary)**

  * Canonical JSON + pipeline for coupling LLM hints with placement policies.
  * Pure text-based netlist encoding strategy optimized for LLM context windows.
  * GRPO adaptations for placement (groupwise advantage normalization, hierarchical actions).
  * Multi-objective normalized reward and surrogate strategies with reproducible pseudocode.

-----

## 2\. Related Work

This project builds on lines of work including parameter-efficient LLM adaptation (LoRA) (Hu et al., 2021), RL for placement (Mirhoseini et al., 2021), and GRPO (Shao et al., 2024). Unlike prior works that rely on graph embeddings for structural understanding, we leverage the advanced reasoning and context capabilities of modern LLMs (like Qwen) to interpret netlist topology directly from textual descriptions. For evaluation and measurement oracles we use open-source CAD toolchains (Yosys, VTR/VPR, OpenROAD).

-----

## 3\. System Overview

Pipeline stages:

1.  Natural language → Verilog (LoRA-adapted **Qwen**; Netlist→Placement pretraining).
2.  Verilog → gate-level netlist / canonical JSON (Yosys/ABC).
3.  Netlist JSON → transformer input: textual tokens → adapter + policy head → placement candidates.
4.  Placement, routing, STA (VPR/OpenROAD) → QoR metrics → GRPO training loop.

A canonical JSON preserves cells, nets, clock domains, simple area estimates and connectivity; it is serialized deterministically for textual input to the LLM.

-----

## 4\. Dataset Construction Workflow

We produce reproducible datasets spanning specification → Verilog → canonical netlist → multiple placement variants → calibrated QoR. Key elements:

  * Specification normalization (functional summary, I/O, timing targets, hints).
  * Verilog generation (human / LLM-assisted / LLM-only) with parse and light semantic checks.
  * Deterministic synthesis to canonical JSON and recording of tool versions/flags/seeds.
  * Two-path placement generation: fast path (many variants, quick metrics) and high-fidelity path (routing + STA for precise labels).
  * Hints, negative examples and augmentations to improve robustness and training diversity.
  * Packaging, netlist-level splits, provenance metadata and automated QC at each stage.

-----

## 5\. Method

The method comprises supervised Netlist→Placement pretraining for **Qwen**, parameter-efficient adapter adaptation (LoRA), GRPO policy, surrogate evaluators, and a multi-objective reward.

### 5.1 Supervised Netlist→Placement Pretraining

Let the supervised dataset be

$$
\begin{align*}
\mathcal{D}_{\text{sup}} &= \{(x_i, y_i)\},
\end{align*}
$$

where ($x_i$) is the deterministic textual serialization of canonical netlist JSON and ($y_i$) is the placement hint sequence. The supervised objective is

$$
\begin{align*}
\mathcal{L}_{\text{sup}}(\theta) &= - \sum_{(x,y)\in\mathcal{D}_{\text{sup}}} \log p_{\theta}(y \mid x),
\end{align*}
$$

with ($p_\theta$) the Qwen token distribution. We adapt via LoRA-style low-rank adapters to keep base weights intact (Hu et al., 2021).

### 5.2 LLM Adaptation: Frozen Qwen, LoRA Adapters, Policy Head

Freeze base Qwen weights ($\theta_0$). Train adapters ($\phi$) and a policy head ($\psi$). The stochastic policy is:

$$
\begin{align*}
\pi_{\phi,\psi}(a \mid s) &= \text{Categorical}\big( f_{\psi}(h_{\phi,\theta_0}(s)) \big),
\end{align*}
$$

where ($h_{\phi,\theta_0}(s)$) is transformer's output on state (s) (pure text tokens of the serialized netlist).

### 5.3 GRPO: Group-normalized Advantages

For each netlist ($N_i$), sample (k) placements ($\{p_{i,j}\}_{j=1}^k$) and obtain scalar rewards ($r_{i,j}$). Let group mean and std be ($\mu_{G_i}$, $\sigma_{G_i}$). The within-group normalized advantage:

$$
\begin{align*}
\widehat{A}_{i,j} &= \frac{r_{i,j} - \mu_{G_i}}{\sigma_{G_i} + \epsilon}.
\end{align*}
$$

The GRPO loss (minimization form) with KL penalty and adapter regularization:

$$
\begin{align*}
\mathcal{L}_{\text{GRPO}}(\phi,\psi)
&= - \mathbb{E}_{i,j}\Big[ \widehat{A}_{i,j} \log \pi_{\phi,\psi}(p_{i,j}\mid N_i) \Big] \\
&\quad + \lambda_{\text{KL}} D_{\mathrm{KL}}\big(\pi_{\text{old}} \parallel \pi_{\phi,\psi}\big) + \lambda_{\phi} \|\phi\|_2^2.
\end{align*}
$$

KL term constrains policy shift (Shao et al., 2024).

### 5.4 Hierarchical Policy Actions

We factorize cell actions into coarse region ($r_c$) and fine tile ($t_c$):

$$
\begin{align*}
\pi(a_c \mid s) &= \pi_{\text{coarse}}(r_c \mid s) \cdot \pi_{\text{fine}}(t_c \mid r_c, s).
\end{align*}
$$

This reduces branching and allows LLM hints to shape coarse placement.

-----

## 6\. Reward 

We use a multi-objective scalar reward:

$$
\begin{align*}
R &= \alpha_{\text{timing}} R_{\text{timing}} + \alpha_{\text{HPWL}} R_{\text{HPWL}} \\
&\quad + \alpha_{\text{cong}} R_{\text{cong}} + \alpha_{\text{overflow}} R_{\text{overflow}} \\
&\quad + \alpha_{\text{power}} R_{\text{power}} - \alpha_{\text{illegal}} R_{\text{illegal}}.
\end{align*}
$$

### 6.1 Timing

Let (WNS) be worst negative slack:

$$
\begin{align*}
R_{\text{timing}} &= -\frac{\max(0, -WNS)}{WNS_{\text{norm}}},
\end{align*}
$$

with calibration constant ($WNS_{\text{norm}}$).

### 6.2 HPWL

Per-net HPWL:

$$
\begin{align*}
hpwl_n &= w_n + h_n,
\end{align*}
$$

Total HPWL:

$$
\begin{align*}
\mathrm{HPWL} &= \sum_n hpwl_n,
\end{align*}
$$

Normalized:

$$
\begin{align*}
R_{\text{HPWL}} &= -\frac{\mathrm{HPWL} - \mathrm{HPWL}_{\min}}{\mathrm{HPWL}_{\text{range}}}.
\end{align*}
$$

### 6.3 Congestion

Tile-level penalty ($c(u_t)$):

$$
\begin{align*}
c(u_t) &=
\begin{cases}
\left(\dfrac{u_t}{u_{\mathrm{cap}}}\right)^{p} & u_t \le u_{\mathrm{cap}}, \\[6pt]
\kappa \cdot \left(\dfrac{u_t}{u_{\mathrm{cap}}}\right)^{p} & u_t > u_{\mathrm{cap}},
\end{cases}
\end{align*}
$$

Global congestion reward:

$$
\begin{align*}
R_{\mathrm{cong}} &= - \frac{ \sum_{t\in\mathcal{T}} w_t \cdot c(u_t) - C_{\min} }{ C_{\mathrm{range}} }.
\end{align*}
$$

Surrogate model predicts ($\hat{u}_t$) for fast evaluation with periodic calibration.

### 6.4 Overflow

Let `overflow` be a nonnegative count:

$$
\begin{align*}
R_{\mathrm{overflow}} &= - \Big[ \beta_{\mathrm{hard}} \cdot \mathbf{1}(\mathrm{overflow} > 0) \\
&\quad + \beta_{\mathrm{soft}} \cdot \frac{\mathrm{overflow}}{O_{\mathrm{norm}}} \Big].
\end{align*}
$$

### 6.5 Power

Wirelength-weighted power proxy:

$$
\begin{align*}
\widehat{P} &= \sum_{n} a_n \cdot (\mathrm{HPWL}_n)^\gamma,
\end{align*}
$$

Normalized:

$$
\begin{align*}
R_{\mathrm{power}} &= - \frac{ \widehat{P} - P_{\min} }{ P_{\mathrm{range}} }.
\end{align*}
$$

### 6.6 Legality

Let ($\mathcal{V}$) be violation set, ($s_v$) severity:

$$
\begin{align*}
R_{\mathrm{illegal}} &= \lambda_{\mathrm{hard}} \cdot \mathbf{1}(|\mathcal{V}|>0) + \sum_{v\in\mathcal{V}} \lambda_v \frac{s_v}{S_v}.
\end{align*}
$$

-----

## 7\. Surrogate Evaluators

Train small regressor surrogates (e.g., efficient MLPs or CNNs) to predict HPWL, slack, overflow and per-tile congestion based on feature maps. Most samples are scored by surrogates; every (M) iterations a stratified subset is routed for full-flow calibration to prevent surrogate exploitation.

-----

## 8\. Training Loop (high-level pseudocode)

**Phases**:

1.  Supervised Netlist→Placement pretraining to initialize adapters.
2.  (Optional) Behavior cloning / imitation learning warm start.
3.  Freeze base Qwen weights; train adapters ($\phi$), policy head ($\psi$) with GRPO; optionally update surrogates.

**Algorithm (textual)**

```
Input: Supervised dataset D_sup, RL dataset D_RL, base Qwen weights theta_0
Initialize adapters phi, policy psi
Phase A: Pretrain on D_sup to minimize L_sup
Phase B: Optional behavior cloning on curated traces
Freeze theta_0
For iteration = 1..N:
  Sample batch of netlists {N_i}
  For each N_i:
    For j = 1..k:
      Build text tokens from N_i
      Form E_in and sample p_{i,j} ~ pi_{phi,psi}(·|E_in)
      Evaluate surrogate reward tilde_r_{i,j}
      Store (N_i, p_{i,j}, tilde_r_{i,j})
  Select subset for full EDA; compute true rewards r_{i,j}
  Calibrate surrogate on new (placement, reward) pairs
  Compute within-group normalized advantages and GRPO loss
  Update phi, psi
```

-----

## 9\. Implementation Notes

  * Synthesis: Yosys + ABC.
  * Fast placement/routing: VTR/VPR.
  * Industrial routing/STA: OpenROAD + OpenROAD-flow-scripts.
  * LLM stack: PyTorch + PEFT/LoRA.
  * GRPO: TRL-style token-output adaptation.
  * Record full provenance (tool versions, flags, seeds) for reproducibility.

-----

## 10\. Conclusion

We present Coder: supervised Netlist→Placement pretraining for **Qwen**, LoRA adapter adaptation, and GRPO-based RL for placement. This preserves the base LLM knowledge while enabling placement-aware adaptation through pure textual understanding of netlists. Future work: stronger surrogate models, careful evaluation of end-to-end RL of adapters vs. base parameters, and large-scale validation with industrial flows.

-----

## How to cite / BibTeX

If you use ideas, data, code, or models from this project, please cite our work and the background works listed below. Put the following BibTeX entries into your `refs.bib` and cite them in your paper.
Any cooperation opportunities, please reach out to me.

> **Cite our work:**

```bibtex
@article{LLMCIRCUIT2025,
  title = {Autonomous Circuit Design: Qwen Netlist-to-Placement Pretraining with GRPO},
  author = {Kailiang Ye},
  year = {2025},
  month={Nov},
  url={\url{[https://github.com/klyebit/LLM_FPGA_Circuit_Design.git]}
}
```

> **Suggested BibTeX entries for referenced background works (edit authors/years/URLs to match real sources):**

```bibtex
@article{bai2023qwen,
  title={Qwen technical report},
  author={Bai, Jinze and Bai, Shuai and Chu, Yunfei and Cui, Zeyu and Dang, Kai and Deng, Xiaodong and Fan, Yang and Ge, Wenbin and Han, Yu and Huang, Fei and others},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}

@article{hu2022lora,
  title={Lora: Low-rank adaptation of large language models.},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu and others},
  journal={ICLR},
  volume={1},
  number={2},
  pages={3},
  year={2022}
}

@article{shao2024deepseekmath,
  title={Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024},
  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, YK and Wu, Y and others},
  journal={URL https://arxiv. org/abs/2402.03300},
  volume={2},
  number={3},
  pages={5},
  year={2024}
}

@article{mirhoseini2020chip,
  title={Chip placement with deep reinforcement learning},
  author={Mirhoseini, Azalia and Goldie, Anna and Yazgan, Mustafa and Jiang, Joe and Songhori, Ebrahim and Wang, Shen and Lee, Young-Joon and Johnson, Eric and Pathak, Omkar and Bae, Sungmin and others},
  journal={arXiv preprint arXiv:2004.10746},
  year={2020}
}

@misc{yosys,
  title = {Yosys Open Synthesis Suite},
  author = {Wolf, C. and contributors},
  year = {2020},
  howpublished = {GitHub},
  note = {https://github.com/YosysHQ/yosys}
}

@misc{vtr_tool,
  title = {VTR / VPR: Versatile Place and Route Tools},
  author = {VTR Developers},
  year = {2025},
  howpublished = {GitHub},
  note = {[https://github.com/VERILOG-TO-ROUTER/vtr-verilog-to-routing](https://docs.verilogtorouting.org/en/latest/vpr/)}
}

@misc{openroad_flow,
  title = {OpenROAD and OpenROAD-flow-scripts},
  author = {OpenROAD developers},
  year = {2025},
  howpublished = {GitHub},
  note = {https://github.com/The-OpenROAD-Project/OpenROAD}
}
```

-----

## Appendix — Notation Cheat Sheet

Displayed important formulas used in the text:

$$
\begin{align*}
\mathcal{D}_{\text{sup}} &= \{(x_i, y_i)\}
\end{align*}
$$

$$
\begin{align*}
\widehat{A}_{i,j} &= \frac{r_{i,j} - \mu_{G_i}}{\sigma_{G_i} + \epsilon}
\end{align*}
$$

$$
\begin{align*}
\mathcal{L}_{\text{GRPO}}(\phi,\psi)
&= - \mathbb{E}_{i,j}\Big[ \widehat{A}_{i,j} \log \pi_{\phi,\psi}(p_{i,j}\mid N_i) \Big] \\
&\quad + \lambda_{\text{KL}} D_{\mathrm{KL}}\big(\pi_{\text{old}} \parallel \pi_{\phi,\psi}\big) + \lambda_{\phi} \|\phi\|_2^2
\end{align*}
$$
