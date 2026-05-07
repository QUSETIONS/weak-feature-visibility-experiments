# 实验证据链索引

本文实验的目标不是堆 benchmark，而是逐层证明一个 detector-class law：

1. matched Gaussian 模型里定律精确成立；
2. GPT-2 activation 背景上 evidence coordinate 仍然有效；
3. 外部自然事件说明这不是注入实验 artifact；
4. SAE-facing audits 说明这个边界对 sparse decoders 有 operational consequence。

## A. Formal synthetic validation

| 编号 | 目的 | 核心结论 | 主要产物 |
|---|---|---|---|
| Server formal v2 | 验证主定理的有限样本后果 | unlabeled slope `-3.954`; labeled slope `-1.902`; evidence-power corr `0.904`; dark-floor corr `0.998` | `results/server_formal_v2/formal_summary.json`; `results/server_formal_v2/*.pdf` |

对应论文位置：
- Main Figure 1
- Experiments: quartic vs quadratic scaling, geometry controls detectability, evidence collapse, dark matter floor

## B. GPT-2 real-activation bridge

| 编号 | 目的 | 核心结论 | 主要产物 |
|---|---|---|---|
| C1/C5 | 收集并 whitening GPT-2 layer-6 residual activations | 100k cache; 30k PCA-whitened cache | `results/real_activation/c1_*`; `results/real_activation/c5_*` |
| C6 | controlled injection evidence collapse | 735 configs; unlabeled power vs `log10(N C_i lambda^4)` corr `0.872`; axis unlabeled null `0.050`, labeled `0.943` | `results/real_activation/c6_detector_evidence_whitened_combined/real_detector_collapse_summary.json`; `real_detector_bridge_main.pdf` |
| C7 | geometry-only intervention | unlabeled power vs geometry corr `0.905`; axis null `0.050`; labeled saturated `1.000` | `results/real_activation/c7_geometry_intervention_whitened/real_geometry_intervention_summary.json` |
| C8 | detector-class stress test | axis feature: labeled `1.000`, fixed-background covariance `0.445`, feature-independent `0.050` | `results/real_activation/c8_assumption_stress_whitened/real_detector_assumption_stress_summary.json` |

对应论文位置：
- Main Figure 2
- Real-activation validation I/II/III

## C. External natural-feature bank

| 编号 | 事件族 | 目的 | 核心结论 | 主要产物 |
|---|---|---|---|---|
| C9 | lexical | 外部词法事件 | 并入 C13 | `results/real_activation/c9_natural_lexical_feature_probe/` |
| C10 | linguistic rules | 规则语言事件 | 并入 C13 | `results/real_activation/c10_natural_linguistic_feature_probe/` |
| C11 | POS tagger | 外部 POS tagger 事件 | 并入 C13 | `results/real_activation/c11_natural_pos_tagger_feature_probe/` |
| C14 | NER | 外部 named-entity 事件 | 并入 C13 | `results/real_activation/c14_natural_ner_feature_probe/` |
| C15 | semantic clusters | text-only cooccurrence clusters | 并入 C13 | `results/real_activation/c15_natural_semantic_cluster_probe/` |
| C13 | preregistered bank | 合并 47 个事件并做多重检验控制 | label-visible `100%`; offdiag `|z|>5` 为 `51.1%`; offdiag BH q<0.05 为 `76.6%`; pooled evidence corr `0.672` | `results/real_activation/c13_preregistered_feature_bank/preregistered_feature_bank_summary.json`; CSV |
| C18 | natural-event geometry audit | 检查自然事件是否落在 axis blind spot | 47 个事件全部远离轴向盲点；geometry min `0.9925`, median `0.9956`; none below `0.99` | `results/real_activation/c18_natural_feature_geometry_audit/` |
| C12 | layer robustness | 排除 layer-6 cherry-picking | GPT-2 layers 4/6/8/10 均 label-visible `100%`; offdiag-visible 至少 `53.8%`; evidence corr >= `0.780` | `results/real_activation/c12_pos_tagger_layer_robustness/pos_layer_robustness_summary.json` |

对应论文位置：
- Real-activation validation III
- Appendix claim-boundary context

## D. SAE-facing audits

| 编号 | 目的 | 核心结论 | 主要产物 |
|---|---|---|---|
| C16 | controlled TopK SAE phase diagram audit | strict recovery ranking AUC by evidence `0.987`; transition-or-recovered AUC `0.948`; diagnostics correlate with evidence around `0.87-0.89` | `results/real_activation/c16_sae_facing_consequence_audit/sae_facing_consequence_audit_summary.json` |
| C17 | pretrained GPT-2 SAE natural-event audit against sparse random dictionary | evidence vs SAE recovery corr `0.379`; evidence vs SAE-over-random gain corr `0.452`; high-evidence tertile gain `+0.066`; overall gain `+0.014`; lexical gain `+0.076` | `results/real_activation/c17_pretrained_sae_natural_event_audit_sparse/` |
| C20/C21 | pretrained GPT-2 SAE layer robustness audit | layers 8 and 4 reproduce evidence ordering: SAE-over-random gain corr `0.339` and `0.539`; high-evidence tertile gains `+0.040` and `+0.086`; low-evidence tertiles remain below random (`-0.024`, `-0.022`) | `results/real_activation/c20_pretrained_sae_layer8_natural_event_audit_sparse/`; `results/real_activation/c21_pretrained_sae_layer4_natural_event_audit_sparse/` |

对应论文位置：
- Appendix: Claim boundary and SAE-facing consequence audit
- Figure `pretrained_sae_natural_event_audit_sparse.pdf`

## E. 建议强调的证据顺序

1. **先讲理论**：二次 vs 四次 + 几何惩罚。
2. **再讲 synthetic**：主定理的 exponent、geometry、evidence coordinate、dark floor 都被验证。
3. **再讲 GPT-2 bridge**：同一 evidence coordinate 在真实 activation 上仍然组织 power。
4. **再讲 natural bank**：47 个外部事件 + FDR + cross-layer，说明不是 cherry-picking。
5. **最后讲 SAE-facing audits**：C16/C17 是 consequence checks，回应 SAE 读者，但不把论文变成 SAE optimization theory。

## F. 已知风险与写法

| 风险 | 不推荐写法 | 推荐写法 |
|---|---|---|
| SAE optimization 过claim | "We explain SAE training failure." | "We prove a local detector-class visibility law relevant to SAE-like decoders." |
| Known-direction detection 过claim | "We solve dictionary recovery." | "Detector experiments isolate feature-level evidence, not end-to-end dictionary identification." |
| Natural events 被认为不够语义 | "Open-ended concept discovery." | "Externally defined natural events and text-derived semantic clusters." |
| Random baseline 审稿口味 | 只报告 SAE atom AUC | 报 C17 sparse-calibrated random dictionary baseline |
| Gaussian abstraction | "This is exactly GPT-2." | "Matched theorem plus GPT-2 bridge evidence." |
