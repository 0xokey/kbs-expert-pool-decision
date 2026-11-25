# Configs

本目录只保留当前主线需要的配置。

## 主配置

```bash
python train.py --config configs/moe_hierarchical_v1.json
```

用途：正式 hard-channel MoE diffusion 训练。

关键设定：

- `model: MoEDiffuser`
- `routing_mode: hard_channel`
- `target_only_diffusion: true`
- `forecast_objective: daily_seasonal_residual`
- `context_len: 168`
- `pred_len: 168`
- `pretrain_epochs: 15`
- `phase1_epochs: 20`
- `phase2_epochs: 30`
- `hard_final_routing: true`
- `final_top_k: 1`
- `eval_sampler: ddim`
- `ddim_t_start: 800`

## Phase 0 通道先验

Phase 0 会先让每个专家在统一 DDPM/DDIM 协议下预训练，然后从训练集内部切出 `prior_calibration` split，使用多窗口、多 diffusion timestep 为每个通道评估专家表现。

相关参数：

- `prior_calibration_ratio: 0.1`：从训练集内部切出的通道先验校准比例。
- `phase0_prior_max_windows: 256`：最多使用多少个 calibration 窗口。
- `phase0_rule_t_values: 5`：每个窗口评估多少个 diffusion timestep。
- `phase0_prior_temperature: 0.1`：根据平均 MSE 生成 soft prior 的温度。
- `phase0_prior_winrate_threshold: 0.6`：最佳专家需要达到的胜率阈值。
- `phase0_prior_margin: 0.05`：最佳专家相对第二名需要达到的损失优势。
- `phase0_prior_hard_mix: 0.7`：证据充分时，将 soft prior 拉向 one-hot 最佳专家的最大比例。

这套逻辑不会把通道直接永久分配给某个专家，而是生成可解释的通道专家先验，再由 Phase 1/2 的 gate 学习确认。

## Debug 配置

```bash
python train.py --config configs/moe_hierarchical_debug.json
```

用途：快速检查数据、模型、训练、评估和导出链路是否能跑通。该配置限制训练窗口和采样步数，指标没有研究意义。

## 保留边界

当前配置比较的是统一 DDPM/DDIM 协议下的异构 denoiser 专家，不是各专家原论文完整协议之间的横向比较。
