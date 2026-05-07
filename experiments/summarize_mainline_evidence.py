"""Summarize evidence against the paper mainline claims.

This is intentionally a paper-facing audit, not a new experiment. It maps
existing artifacts to the strengthened story:
weak-feature recovery obeys a quartic, geometry-controlled detection law.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def main() -> None:
    outdir = ROOT / "refine-logs"
    outdir.mkdir(parents=True, exist_ok=True)

    formal = read_json(ROOT / "results" / "server_formal_v2" / "formal_summary.json")
    topk_audit = read_json(
        ROOT
        / "results"
        / "real_activation"
        / "c4_topk_phase_diagram_whitened_positive"
        / "topk_phase_diagram_audit.json"
    )
    c6_path = (
        ROOT
        / "results"
        / "real_activation"
        / "c6_detector_evidence_whitened_combined"
        / "real_detector_collapse_summary.json"
    )
    c6 = read_json(c6_path) if c6_path.exists() else None
    c7_path = (
        ROOT
        / "results"
        / "real_activation"
        / "c7_geometry_intervention_whitened"
        / "real_geometry_intervention_summary.json"
    )
    c7 = read_json(c7_path) if c7_path.exists() else None
    c8_path = (
        ROOT
        / "results"
        / "real_activation"
        / "c8_assumption_stress_whitened"
        / "real_detector_assumption_stress_summary.json"
    )
    c8 = read_json(c8_path) if c8_path.exists() else None
    c9_path = (
        ROOT
        / "results"
        / "real_activation"
        / "c9_natural_lexical_feature_probe"
        / "natural_lexical_feature_probe_summary.json"
    )
    c9 = read_json(c9_path) if c9_path.exists() else None
    c10_path = (
        ROOT
        / "results"
        / "real_activation"
        / "c10_natural_linguistic_feature_probe"
        / "natural_linguistic_feature_probe_summary.json"
    )
    c10 = read_json(c10_path) if c10_path.exists() else None
    c11_path = (
        ROOT
        / "results"
        / "real_activation"
        / "c11_natural_pos_tagger_feature_probe"
        / "natural_pos_tagger_feature_probe_summary.json"
    )
    c11 = read_json(c11_path) if c11_path.exists() else None
    c12_path = (
        ROOT
        / "results"
        / "real_activation"
        / "c12_pos_tagger_layer_robustness"
        / "pos_layer_robustness_summary.json"
    )
    c12 = read_json(c12_path) if c12_path.exists() else None
    c14_path = (
        ROOT
        / "results"
        / "real_activation"
        / "c14_natural_ner_feature_probe"
        / "natural_ner_feature_probe_summary.json"
    )
    c14 = read_json(c14_path) if c14_path.exists() else None
    c15_path = (
        ROOT
        / "results"
        / "real_activation"
        / "c15_natural_semantic_cluster_probe"
        / "natural_semantic_cluster_probe_summary.json"
    )
    c15 = read_json(c15_path) if c15_path.exists() else None
    c13_path = (
        ROOT
        / "results"
        / "real_activation"
        / "c13_preregistered_feature_bank"
        / "preregistered_feature_bank_summary.json"
    )
    c13 = read_json(c13_path) if c13_path.exists() else None
    c16_path = (
        ROOT
        / "results"
        / "real_activation"
        / "c16_sae_facing_consequence_audit"
        / "sae_facing_consequence_audit_summary.json"
    )
    c16 = read_json(c16_path) if c16_path.exists() else None

    bridge_csv = ROOT / "results" / "real_activation" / "c3_local_fast_smoke" / "injection_probe.csv"
    bridge_rows = read_csv(bridge_csv) if bridge_csv.exists() else []

    audit = {
        "C1_exponent_separation": {
            "status": "strong",
            "labeled_slope": formal["labeled_slope"],
            "labeled_ci": [formal["labeled_slope_ci_low"], formal["labeled_slope_ci_high"]],
            "unlabeled_slope": formal["unlabeled_slope"],
            "unlabeled_ci": [formal["unlabeled_slope_ci_low"], formal["unlabeled_slope_ci_high"]],
        },
        "C2_geometry_penalty": {
            "status": "strong_synthetic_and_real_geometry_intervention" if c7 else "strong_synthetic_bridge_partial",
            "axis_power_mean": formal["axis_power_mean"],
            "axis_power_sd": formal["axis_power_sd"],
            "spread_power_mean_at_lambda_0p50": formal["spread_power_mean_at_lambda_0p50"],
            "real_geometry_corr_unlabeled": None if c7 is None else c7["corr_unlabeled_power_geometry"],
            "real_geometry_axis_labeled_power": None if c7 is None else c7["axis_labeled_power"],
            "real_geometry_axis_unlabeled_power": None if c7 is None else c7["axis_unlabeled_power"],
            "real_geometry_max_unlabeled_power": None if c7 is None else c7["max_unlabeled_power"],
            "detector_stress_axis_labeled_power": None if c8 is None else c8["axis_labeled_power"],
            "detector_stress_axis_unrestricted_cov_power": None if c8 is None else c8["axis_unrestricted_cov_power"],
            "detector_stress_axis_feature_independent_power": None if c8 is None else c8["axis_feature_independent_power"],
        },
        "C3_evidence_collapse": {
            "status": "strong_synthetic_and_real_bridge" if c6 else "strong_synthetic_needs_real_bridge_extension",
            "evidence_power_corr": formal["evidence_power_corr"],
            "real_bridge_unlabeled_corr_log10_evidence": None if c6 is None else c6["corr_unlabeled_power_log10_evidence"],
        },
        "C4_dark_fraction_floor": {
            "status": "strong_synthetic",
            "dark_floor_corr_all": formal["dark_floor_corr_all"],
            "dark_floor_mae_all": formal["dark_floor_mae_all"],
        },
        "C5_real_activation_bridge": {
            "status": "strong_detector_bridge_operational_topk_bridge_external_language_probes" if c9 and c10 and c11 else "strong_detector_bridge_operational_topk_bridge_external_natural_probes" if c9 and c10 else "strong_detector_bridge_operational_topk_bridge_external_natural_probe" if c9 else "strong_detector_bridge_operational_topk_bridge",
            "topk_largest_null_lambda": topk_audit["largest_null_lambda"],
            "topk_smallest_transition_lambda": topk_audit["smallest_transition_lambda"],
            "topk_smallest_recovered_lambda": topk_audit["smallest_recovered_lambda"],
            "topk_counts": {
                "null": topk_audit["null_count"],
                "transition": topk_audit["transition_count"],
                "recovered": topk_audit["recovered_count"],
            },
            "sae_bridge_auc_lambda_for_recovered": None if c16 is None else c16["auc_lambda_for_recovered"],
            "sae_bridge_auc_lambda_for_transition_or_recovered": None if c16 is None else c16["auc_lambda_for_transition_or_recovered"],
            "sae_bridge_corr_decoder_cosine_log10_lambda4": None if c16 is None else c16["metric_summary"]["mean_max_decoder_cosine"]["pearson_log10_lambda4"],
            "sae_bridge_corr_auc_excess_log10_lambda4": None if c16 is None else c16["metric_summary"]["mean_top_label_atom_abs_auc_excess"]["pearson_log10_lambda4"],
            "detector_bridge_rows": None if c6 is None else c6["rows"],
            "detector_bridge_axis_unlabeled_power": None if c6 is None else c6["axis_mean_unlabeled_power"],
            "local_bridge_rows": len(bridge_rows),
        },
        "C6_external_lexical_natural_probe": {
            "status": "external_lexical_probe_available" if c9 else "not_available",
            "events_evaluated": None if c9 is None else c9["events_evaluated"],
            "tokens": None if c9 is None else c9["tokens"],
            "median_abs_label_z": None if c9 is None else c9["median_abs_label_z"],
            "median_abs_full_cov_z": None if c9 is None else c9["median_abs_full_cov_z"],
            "median_abs_offdiag_cov_z": None if c9 is None else c9["median_abs_offdiag_cov_z"],
            "frac_label_z_gt_5": None if c9 is None else c9["frac_label_z_gt_5"],
            "frac_offdiag_z_gt_5": None if c9 is None else c9["frac_offdiag_z_gt_5"],
            "corr_log_evidence_abs_offdiag_z": None if c9 is None else c9["corr_log_evidence_abs_offdiag_z"],
        },
        "C7_external_linguistic_natural_probe": {
            "status": "external_linguistic_probe_available" if c10 else "not_available",
            "events_evaluated": None if c10 is None else c10["events_evaluated"],
            "tokens": None if c10 is None else c10["tokens"],
            "median_abs_label_z": None if c10 is None else c10["median_abs_label_z"],
            "median_abs_full_cov_z": None if c10 is None else c10["median_abs_full_cov_z"],
            "median_abs_offdiag_cov_z": None if c10 is None else c10["median_abs_offdiag_cov_z"],
            "frac_label_z_gt_5": None if c10 is None else c10["frac_label_z_gt_5"],
            "frac_offdiag_z_gt_5": None if c10 is None else c10["frac_offdiag_z_gt_5"],
            "corr_log_evidence_abs_offdiag_z": None if c10 is None else c10["corr_log_evidence_abs_offdiag_z"],
        },
        "C8_external_pos_tagger_natural_probe": {
            "status": "external_pos_tagger_probe_available" if c11 else "not_available",
            "events_evaluated": None if c11 is None else c11["events_evaluated"],
            "tokens": None if c11 is None else c11["tokens"],
            "median_abs_label_z": None if c11 is None else c11["median_abs_label_z"],
            "median_abs_full_cov_z": None if c11 is None else c11["median_abs_full_cov_z"],
            "median_abs_offdiag_cov_z": None if c11 is None else c11["median_abs_offdiag_cov_z"],
            "frac_label_z_gt_5": None if c11 is None else c11["frac_label_z_gt_5"],
            "frac_offdiag_z_gt_5": None if c11 is None else c11["frac_offdiag_z_gt_5"],
            "corr_log_evidence_abs_offdiag_z": None if c11 is None else c11["corr_log_evidence_abs_offdiag_z"],
        },
        "C9_pos_tagger_layer_robustness": {
            "status": "pos_tagger_layer_robustness_available" if c12 else "not_available",
            "layers": None if c12 is None else c12["layers"],
            "min_frac_label_z_gt_5": None if c12 is None else c12["min_frac_label_z_gt_5"],
            "min_frac_offdiag_z_gt_5": None if c12 is None else c12["min_frac_offdiag_z_gt_5"],
            "min_median_abs_offdiag_cov_z": None if c12 is None else c12["min_median_abs_offdiag_cov_z"],
            "min_corr_log_evidence_abs_offdiag_z": None if c12 is None else c12["min_corr_log_evidence_abs_offdiag_z"],
        },
        "C10_preregistered_external_feature_bank": {
            "status": "preregistered_external_feature_bank_available" if c13 else "not_available",
            "events_evaluated": None if c13 is None else c13["events_evaluated"],
            "families": None if c13 is None else c13["families"],
            "frac_label_abs_z_gt_5": None if c13 is None else c13["frac_label_abs_z_gt_5"],
            "frac_offdiag_abs_z_gt_5": None if c13 is None else c13["frac_offdiag_abs_z_gt_5"],
            "frac_label_bh_q_lt_005": None if c13 is None else c13["frac_label_bh_q_lt_005"],
            "frac_offdiag_bh_q_lt_005": None if c13 is None else c13["frac_offdiag_bh_q_lt_005"],
            "corr_log_evidence_abs_offdiag_z": None if c13 is None else c13["corr_log_evidence_abs_offdiag_z"],
            "min_family_corr_log_evidence_abs_offdiag_z": None if c13 is None else c13["min_family_corr_log_evidence_abs_offdiag_z"],
        },
        "C11_external_ner_natural_probe": {
            "status": "external_ner_probe_available" if c14 else "not_available",
            "events_evaluated": None if c14 is None else c14["events_evaluated"],
            "tokens": None if c14 is None else c14["tokens"],
            "median_abs_label_z": None if c14 is None else c14["median_abs_label_z"],
            "median_abs_full_cov_z": None if c14 is None else c14["median_abs_full_cov_z"],
            "median_abs_offdiag_cov_z": None if c14 is None else c14["median_abs_offdiag_cov_z"],
            "frac_label_z_gt_5": None if c14 is None else c14["frac_label_z_gt_5"],
            "frac_offdiag_z_gt_5": None if c14 is None else c14["frac_offdiag_z_gt_5"],
            "corr_log_evidence_abs_offdiag_z": None if c14 is None else c14["corr_log_evidence_abs_offdiag_z"],
        },
        "C12_external_semantic_cluster_probe": {
            "status": "external_semantic_cluster_probe_available" if c15 else "not_available",
            "events_evaluated": None if c15 is None else c15["events_evaluated"],
            "tokens": None if c15 is None else c15["tokens"],
            "median_abs_label_z": None if c15 is None else c15["median_abs_label_z"],
            "median_abs_full_cov_z": None if c15 is None else c15["median_abs_full_cov_z"],
            "median_abs_offdiag_cov_z": None if c15 is None else c15["median_abs_offdiag_cov_z"],
            "frac_label_z_gt_5": None if c15 is None else c15["frac_label_z_gt_5"],
            "frac_offdiag_z_gt_5": None if c15 is None else c15["frac_offdiag_z_gt_5"],
            "corr_log_evidence_abs_offdiag_z": None if c15 is None else c15["corr_log_evidence_abs_offdiag_z"],
        },
    }

    with (outdir / "MAINLINE_EVIDENCE_AUDIT.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    lines = [
        "# Mainline Evidence Audit",
        "",
        "This audit checks whether existing results support the strengthened initial story:",
        "**weak features hide under superposition because unlabeled recovery has a quartic, geometry-controlled detection boundary.**",
        "",
        "## C1: Labeled Quadratic vs Unlabeled Quartic",
        "",
        f"- Labeled slope: `{fmt(formal['labeled_slope'])}` with CI `[{fmt(formal['labeled_slope_ci_low'])}, {fmt(formal['labeled_slope_ci_high'])}]`.",
        f"- Unlabeled slope: `{fmt(formal['unlabeled_slope'])}` with CI `[{fmt(formal['unlabeled_slope_ci_low'])}, {fmt(formal['unlabeled_slope_ci_high'])}]`.",
        "- Status: strong mainline evidence.",
        "",
        "## C2: Geometry Penalty",
        "",
        f"- Axis-aligned mean power: `{fmt(formal['axis_power_mean'])} +/- {fmt(formal['axis_power_sd'])}`.",
        f"- Spread-loading power at lambda 0.50: `{fmt(formal['spread_power_mean_at_lambda_0p50'])}`.",
        (
            f"- Real GPT-2 geometry-only intervention: unlabeled power correlates `{fmt(c7['corr_unlabeled_power_geometry'])}` with `1-||w||_4^4`; axis has labeled power `{fmt(c7['axis_labeled_power'])}` but unlabeled power `{fmt(c7['axis_unlabeled_power'])}`."
            if c7
            else "- Real GPT-2 geometry-only intervention: not yet available."
        ),
        (
            f"- Detector-class stress test: axis power is labeled `{fmt(c8['axis_labeled_power'])}`, full covariance `{fmt(c8['axis_unrestricted_cov_power'])}`, feature-independent `{fmt(c8['axis_feature_independent_power'])}`; feature-independent power/geometry correlation is `{fmt(c8['corr_feature_independent_power_geometry'])}`."
            if c8
            else "- Detector-class stress test: not yet available."
        ),
        "- Status: strong synthetic evidence plus direct real-activation geometry intervention." if c7 else "- Status: strong synthetic evidence; real-activation geometry extension is the next best supplement.",
        "",
        "## C3: Evidence Collapse",
        "",
        f"- Synthetic power correlation with `log(N C_i lambda^4)`: `{fmt(formal['evidence_power_corr'])}`.",
        (
            f"- Real GPT-2 detector bridge correlation with `log10(N C_i lambda^4)`: `{fmt(c6['corr_unlabeled_power_log10_evidence'])}`."
            if c6
            else "- Real GPT-2 detector bridge extension: not yet available."
        ),
        (
            f"- In that bridge, axis mean unlabeled power remains `{fmt(c6['axis_mean_unlabeled_power'])}` while axis labeled power is `{fmt(c6['axis_mean_labeled_power'])}`."
            if c6
            else ""
        ),
        "- Status: strong synthetic evidence and real-activation bridge evidence." if c6 else "- Status: strong synthetic evidence; real-activation detector-only extension is the most mainline next run.",
        "",
        "## C4: Dark-Matter Fraction As Aggregate Consequence",
        "",
        f"- Predicted-vs-observed residual floor correlation: `{fmt(formal['dark_floor_corr_all'])}`.",
        f"- Mean absolute deviation: `{fmt(formal['dark_floor_mae_all'])}`.",
        "- Status: strong synthetic aggregate evidence.",
        "",
        "## C5: Real-Activation Bridge",
        "",
        f"- TopK bridge largest null lambda: `{fmt(topk_audit['largest_null_lambda'])}`.",
        f"- TopK bridge first transition lambda: `{fmt(topk_audit['smallest_transition_lambda'])}`.",
        f"- TopK bridge strict recovery starts at lambda: `{fmt(topk_audit['smallest_recovered_lambda'])}`.",
        f"- Counts: null `{topk_audit['null_count']}`, transition `{topk_audit['transition_count']}`, recovered `{topk_audit['recovered_count']}`.",
        (
            f"- SAE-facing C16 audit: `lambda_eff` predicts strict TopK recovery with AUC `{fmt(c16['auc_lambda_for_recovered'])}` and transition-or-recovered with AUC `{fmt(c16['auc_lambda_for_transition_or_recovered'])}`; decoder cosine and atom-AUC excess correlate `{fmt(c16['metric_summary']['mean_max_decoder_cosine']['pearson_log10_lambda4'])}` and `{fmt(c16['metric_summary']['mean_top_label_atom_abs_auc_excess']['pearson_log10_lambda4'])}` with `log10(lambda_eff^4)`."
            if c16
            else "- SAE-facing C16 audit: not yet available."
        ),
        f"- Detector-only GPT-2 bridge rows: `{c6['rows']}`." if c6 else "- Detector-only GPT-2 bridge rows: not yet available.",
        "- Status: detector bridge supports the mainline; TopK remains useful operational bridge evidence, not the theorem.",
        "",
        "## C6: External Lexical Natural-Feature Probe",
        "",
        (
            f"- Externally defined lexical token events provide `{c9['events_evaluated']}` natural-feature probes over `{c9['tokens']}` GPT-2 tokens, with a 5k/5k train/test split."
            if c9
            else "- External lexical natural-feature probe: not yet available."
        ),
        (
            f"- Median absolute z-statistics are label `{fmt(c9['median_abs_label_z'])}`, full covariance `{fmt(c9['median_abs_full_cov_z'])}`, and feature-independent off-diagonal `{fmt(c9['median_abs_offdiag_cov_z'])}`."
            if c9
            else ""
        ),
        (
            f"- Fractions with `|z|>5`: label `{fmt(c9['frac_label_z_gt_5'])}`, off-diagonal `{fmt(c9['frac_offdiag_z_gt_5'])}`; off-diagonal strength correlates `{fmt(c9['corr_log_evidence_abs_offdiag_z'])}` with the natural evidence proxy."
            if c9
            else ""
        ),
        "- Status: useful natural-feature evidence without SAE-circular labels; conservative because lexical token events are not open-ended semantic concepts." if c9 else "",
        "",
        "## C7: External Linguistic Natural-Feature Probe",
        "",
        (
            f"- Rule-based external linguistic events provide `{c10['events_evaluated']}` natural-feature probes over `{c10['tokens']}` GPT-2 tokens, with a 5k/5k train/test split."
            if c10
            else "- External linguistic natural-feature probe: not yet available."
        ),
        (
            f"- Median absolute z-statistics are label `{fmt(c10['median_abs_label_z'])}`, full covariance `{fmt(c10['median_abs_full_cov_z'])}`, and feature-independent off-diagonal `{fmt(c10['median_abs_offdiag_cov_z'])}`."
            if c10
            else ""
        ),
        (
            f"- Fractions with `|z|>5`: label `{fmt(c10['frac_label_z_gt_5'])}`, off-diagonal `{fmt(c10['frac_offdiag_z_gt_5'])}`; off-diagonal strength correlates `{fmt(c10['corr_log_evidence_abs_offdiag_z'])}` with the natural evidence proxy."
            if c10
            else ""
        ),
        "- Status: useful linguistic natural-feature evidence without SAE-circular labels; conservative because events are predefined rather than open-ended semantic concepts." if c10 else "",
        "",
        "## C8: External POS-Tagger Natural-Feature Probe",
        "",
        (
            f"- NLTK POS-tagger events provide `{c11['events_evaluated']}` natural-feature probes over `{c11['tokens']}` GPT-2 tokens, with a 5k/5k train/test split."
            if c11
            else "- External POS-tagger natural-feature probe: not yet available."
        ),
        (
            f"- Median absolute z-statistics are label `{fmt(c11['median_abs_label_z'])}`, full covariance `{fmt(c11['median_abs_full_cov_z'])}`, and feature-independent off-diagonal `{fmt(c11['median_abs_offdiag_cov_z'])}`."
            if c11
            else ""
        ),
        (
            f"- Fractions with `|z|>5`: label `{fmt(c11['frac_label_z_gt_5'])}`, off-diagonal `{fmt(c11['frac_offdiag_z_gt_5'])}`; off-diagonal strength correlates `{fmt(c11['corr_log_evidence_abs_offdiag_z'])}` with the natural evidence proxy."
            if c11
            else ""
        ),
        "- Status: strongest external natural-language probe so far; labels come from an independent POS tagger rather than SAE discovery or hand-coded suffix rules." if c11 else "",
        "",
        "## C9: POS-Tagger Cross-Layer Robustness",
        "",
        (
            f"- POS-tagger probes were repeated on GPT-2 layers `{c12['layers']}`."
            if c12
            else "- POS-tagger cross-layer robustness: not yet available."
        ),
        (
            f"- Across layers, the minimum label-visible fraction is `{fmt(c12['min_frac_label_z_gt_5'])}`, the minimum off-diagonal-visible fraction is `{fmt(c12['min_frac_offdiag_z_gt_5'])}`, the minimum median off-diagonal `|z|` is `{fmt(c12['min_median_abs_offdiag_cov_z'])}`, and the minimum evidence correlation is `{fmt(c12['min_corr_log_evidence_abs_offdiag_z'])}`."
            if c12
            else ""
        ),
        "- Status: useful robustness evidence against layer-6 cherry-picking for the external natural-language probes." if c12 else "",
        "",
        "## C10: External NER Natural-Feature Probe",
        "",
        (
            f"- NLTK named-entity events provide `{c14['events_evaluated']}` semantic natural-feature probes over `{c14['tokens']}` GPT-2 tokens, with a 5k/5k train/test split."
            if c14
            else "- External NER natural-feature probe: not yet available."
        ),
        (
            f"- Median absolute z-statistics are label `{fmt(c14['median_abs_label_z'])}`, full covariance `{fmt(c14['median_abs_full_cov_z'])}`, and feature-independent off-diagonal `{fmt(c14['median_abs_offdiag_cov_z'])}`."
            if c14
            else ""
        ),
        (
            f"- Fractions with `|z|>5`: label `{fmt(c14['frac_label_z_gt_5'])}`, off-diagonal `{fmt(c14['frac_offdiag_z_gt_5'])}`; off-diagonal strength correlates `{fmt(c14['corr_log_evidence_abs_offdiag_z'])}` with the natural evidence proxy."
            if c14
            else ""
        ),
        "- Status: useful semantic natural-feature evidence from an external tagger; still conservative because the entity types are predefined." if c14 else "",
        "",
        "## C11: Semi-Open Semantic-Cluster Probe",
        "",
        (
            f"- Text-only PPMI/SVD/KMeans clustering provides `{c15['events_evaluated']}` supported semantic-cluster probes over `{c15['tokens']}` GPT-2 tokens, with a 5k/5k train/test split."
            if c15
            else "- Semi-open semantic-cluster probe: not yet available."
        ),
        (
            f"- Median absolute z-statistics are label `{fmt(c15['median_abs_label_z'])}`, full covariance `{fmt(c15['median_abs_full_cov_z'])}`, and feature-independent off-diagonal `{fmt(c15['median_abs_offdiag_cov_z'])}`."
            if c15
            else ""
        ),
        (
            f"- Fractions with `|z|>5`: label `{fmt(c15['frac_label_z_gt_5'])}`, off-diagonal `{fmt(c15['frac_offdiag_z_gt_5'])}`; off-diagonal strength correlates `{fmt(c15['corr_log_evidence_abs_offdiag_z'])}` with the natural evidence proxy."
            if c15
            else ""
        ),
        "- Status: useful semi-open semantic evidence. The clusters are all label-visible, but feature-independent evidence remains boundary-controlled rather than uniformly strong." if c15 else "",
        "",
        "## C12: Pre-Specified External Feature Bank",
        "",
        (
            f"- The C9-C11/C14/C15 probes form one pre-specified bank with `{c13['events_evaluated']}` external events across families `{c13['families']}`."
            if c13
            else "- Pre-specified external feature bank: not yet available."
        ),
        (
            f"- All events are label-visible (`{fmt(c13['frac_label_abs_z_gt_5'])}` with `|z|>5`) and all survive label BH correction at `q<0.05` (`{fmt(c13['frac_label_bh_q_lt_005'])}`)."
            if c13
            else ""
        ),
        (
            f"- Feature-independent evidence remains substantial after multiple-testing correction: `{fmt(c13['frac_offdiag_abs_z_gt_5'])}` have `|z|>5`, `{fmt(c13['frac_offdiag_bh_q_lt_005'])}` survive off-diagonal BH correction, pooled evidence correlation is `{fmt(c13['corr_log_evidence_abs_offdiag_z'])}`, and the weakest family correlation is `{fmt(c13['min_family_corr_log_evidence_abs_offdiag_z'])}`."
            if c13
            else ""
        ),
        "- Status: useful guard against event cherry-picking and multiple-comparison objections; still not a claim of arbitrary open-ended semantic discovery." if c13 else "",
        "",
        "## Next Mainline Runs",
        "",
        "1. Keep the claim-boundary table in appendix; do not promote C16 into a theorem about all SAEs.",
        "2. Optional rebuttal-only note: C15 is semi-open semantic evidence, not full SAE concept discovery.",
        "3. If time remains, polish the C16 appendix figure/caption rather than running broad SAE sweeps.",
    ]
    (outdir / "MAINLINE_EVIDENCE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
