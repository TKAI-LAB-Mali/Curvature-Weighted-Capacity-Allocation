"""
plot_norms.py
=============
Visualise how weight and activation norms change after pruning, across the
5 MDL ratio sets and 3 pruning methods.

Usage
-----
    python plot_norms.py \
        --results_root results/ \
        --output_dir plots/ \
        --ratio_paths /path/to/cola.json /path/to/mrpc.json ...

Directory layout expected (produced by main.py --track_norms):

    results/
      {MODEL_NAME}_MDL_<ratio_name>/
        <ratio_name>.json            <- copied by the bash script
        <method>/
          weight_norms_before.json
          weight_norms_after.json
          activation_norms_before.json
          activation_norms_after.json

Output layout (mirrors args.save from main.py):

    results/
      {MODEL_NAME}_MDL_<ratio_name>/
        weight_norm_method_overlay.png      <- spans all methods
        activation_norm_method_overlay.png
        <method>/
          weight_norm_before_after.png      <- per method
          activation_norm_before_after.png
          weight_norm_interactive.html

    plots/   (--output_dir, for plots that span all ratio sets)
      weight_norm_ratio_overlay_<method>.png
      activation_norm_ratio_overlay_<method>.png
      weight_norm_delta_heatmap.png
      activation_norm_delta_heatmap.png
"""

import sys
import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from utils import aggregate_weight_norms_from_dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME="Mistral-7B-v0.1"
METHODS = ["sparsegpt", "magnitude", "wanda"]
KNOWN_DATASETS = ["cola", "mrpc", "openbook", "text_science_q_rebuttal", "commonq"]
METHOD_COLORS = {"sparsegpt": "#E63946", "magnitude": "#2A9D8F", "wanda": "#E9C46A"}
BEFORE_COLOR = "#AAAAAA"
RATIO_COLOR  = "#7B68EE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_ratio_path(ratio_path):
    """
    Load an MDL sparsity ratio JSON from an explicit file path supplied by
    the caller (e.g. passed from the bash script via --ratio_paths).

    Returns {layer_idx: sparsity_ratio} or {} if the path is None,
    does not exist, or does not contain a "rho_l" key.
    """
    if ratio_path is None:
        return {}
    p = Path(ratio_path)
    if not p.exists():
        print(f"  [warn] ratio file not found: {p}")
        return {}
    data = load_json(p)
    if "rho_l" not in data:
        print(f"  [warn] no 'rho_l' key in {p}")
        return {}
    return {i: float(v) for i, v in enumerate(data["rho_l"])}


def load_experiment(exp_dir, ratio_name, method):
    """
    Load before/after norm dicts from a single method directory.

    Returns a tuple of six dicts:
        (agg_w_before, agg_w_after, act_before, act_after,
         raw_w_before, raw_w_after)

    agg_*   -- {layer_idx: scalar}  aggregated L2 norm per layer
    act_*   -- {layer_idx: scalar}  activation norm per layer
    raw_w_* -- {layer_idx: {submodule: scalar}}  per-submodule norms,
               used by plot_aggregated_weight_norms for the secondary panel

    Returns None if any required file is missing.
    """
    needed = [
        f"weight_norms_before_{method}_{ratio_name}.json", f"weight_norms_after_{method}_{ratio_name}.json",
        f"activation_norms_before_{method}_{ratio_name}.json", f"activation_norms_after_{method}_{ratio_name}.json",
    ]
    if not all((exp_dir / f).exists() for f in needed):
        return None

    raw_w_before_json = load_json(exp_dir / f"weight_norms_before_{method}_{ratio_name}.json")
    raw_w_after_json  = load_json(exp_dir / f"weight_norms_after_{method}_{ratio_name}.json")

    agg_w_before = aggregate_weight_norms_from_dict(raw_w_before_json)
    agg_w_after  = aggregate_weight_norms_from_dict(raw_w_after_json)

    act_before = {int(k): float(v) for k, v in load_json(exp_dir / f"activation_norms_before_{method}_{ratio_name}.json").items()}
    act_after  = {int(k): float(v) for k, v in load_json(exp_dir / f"activation_norms_after_{method}_{ratio_name}.json").items()}

    raw_w_before = {int(k): v for k, v in raw_w_before_json.items()}
    raw_w_after  = {int(k): v for k, v in raw_w_after_json.items()}

    return agg_w_before, agg_w_after, act_before, act_after, raw_w_before, raw_w_after


def common_layers(*dicts):
    """Return sorted list of layer indices present in every dict."""
    sets = [set(d.keys()) for d in dicts]
    return sorted(set.intersection(*sets))



def get_common_suffix(strings):
    """
    Finds the longest common suffix among a list of strings.
    E.g., ['cola_mdl_minmax', 'mrpc_mdl_minmax'] -> 'mdl_minmax'
    """
    if not strings:
        return "default_run"
    
    # Reverse all strings to use commonprefix
    reversed_strings = [s[::-1] for s in strings]
    common_rev = os.path.commonprefix(reversed_strings)
    
    # Reverse back and strip any leading underscores
    common_suffix = common_rev[::-1].strip("_")
    
    return common_suffix if common_suffix else "global_summary"

def extract_family_suffix(ratio_name):
    """
    Strips the dataset prefix to find the underlying experiment family.
    E.g., 'cola_mdl_minmax_0.55' -> 'mdl_minmax_0.55'
    """
    for ds in KNOWN_DATASETS:
        if ratio_name.startswith(f"{ds}_"):
            return ratio_name.removeprefix(f"{ds}_")
    return "default_run"

# ---------------------------------------------------------------------------
# Per-experiment plot: before vs after (saved in method subdir)
# ---------------------------------------------------------------------------

def plot_before_after(layers, before, after, method, ratio_name, norm_type,
                      save_dir, sparsity_ratios=None):
    """
    Static PNG with two panels:
      Left  — before/after norm lines with optional sparsity ratio twin axis
      Right — % norm drop per layer as horizontal bars

    Saved to save_dir (the method subdir), which already encodes both
    ratio_name and method, so the filename itself is kept short.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]})
    ax_line, ax_bar = axes

    before_vals = [before[l] for l in layers]
    after_vals  = [after[l]  for l in layers]

    # --- Line plot ---
    ax_line.plot(layers, before_vals, color=BEFORE_COLOR, linewidth=1.5,
                 linestyle="--", label="Before pruning", zorder=2)
    ax_line.plot(layers, after_vals, color=METHOD_COLORS.get(method, "#333"),
                 linewidth=2, label=f"After ({method})", zorder=3)
    ax_line.fill_between(layers, before_vals, after_vals,
                         alpha=0.12, color=METHOD_COLORS.get(method, "#333"))
    ax_line.set_xlabel("Transformer layer index", fontsize=11)
    ax_line.set_ylabel("L2 norm", fontsize=11)
    ax_line.set_title(
        f"{norm_type.replace('_', ' ').title()}  |  {ratio_name}  |  {method}",
        fontsize=12, fontweight="bold",
    )
    ax_line.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax_line.grid(axis="y", alpha=0.3)

    # --- Sparsity ratio twin axis ---
    # Ratios live on [0, 1] while norms can be orders of magnitude larger,
    # so a secondary y-axis keeps both readable without scale distortion.
    if sparsity_ratios:
        ratio_layers = [l for l in layers if l in sparsity_ratios]
        ratio_vals   = [sparsity_ratios[l] for l in ratio_layers]
        ax_twin = ax_line.twinx()
        ax_twin.step(ratio_layers, ratio_vals, where="mid",
                     color=RATIO_COLOR, linewidth=1.2, linestyle=":",
                     alpha=0.75, label="Sparsity ratio (rho_l)", zorder=1)
        ax_twin.set_ylabel("Sparsity ratio", fontsize=10, color=RATIO_COLOR)
        ax_twin.tick_params(axis="y", labelcolor=RATIO_COLOR)
        ax_twin.set_ylim(0, 1.05)
        lines_main, labels_main = ax_line.get_legend_handles_labels()
        lines_twin, labels_twin = ax_twin.get_legend_handles_labels()
        ax_line.legend(lines_main + lines_twin, labels_main + labels_twin,
                       framealpha=0.9, fontsize=9)
    else:
        ax_line.legend(framealpha=0.9)

    # --- Bar: relative drop per layer ---
    drops = [100 * (b - a) / b if b != 0 else 0 for b, a in zip(before_vals, after_vals)]
    bar_colors = [METHOD_COLORS.get(method, "#333") if d >= 0 else "#FF8FA3" for d in drops]
    ax_bar.barh(layers, drops, color=bar_colors, height=0.8)
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_xlabel("Norm drop (%)", fontsize=10)
    ax_bar.set_title("Δ per layer", fontsize=10)
    ax_bar.yaxis.set_major_locator(ticker.MultipleLocator(8))
    ax_bar.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out = save_dir / f"{norm_type}_before_after_{method}_{ratio_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Interactive weight norm plot with per-submodule secondary panel
# (saved in method subdir)
# ---------------------------------------------------------------------------

def plot_aggregated_weight_norms(layers, agg_before, agg_after,
                                  raw_before, raw_after,
                                  method, ratio_name, save_dir,
                                  sparsity_ratios=None):
    """
    Interactive plotly HTML: top panel shows aggregated L2 norm per layer
    (before/after) with optional sparsity ratio on a secondary y-axis.
    Bottom panel shows the per-submodule breakdown for whichever layer is
    selected from a dropdown.

    Saved to save_dir (the method subdir).
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  [skip] plotly not installed — run: pip install plotly")
        return

    method_color = METHOD_COLORS.get(method, "#333333")

    before_vals = [agg_before[l] for l in layers]
    after_vals  = [agg_after[l]  for l in layers]

    specs = [[{"secondary_y": bool(sparsity_ratios)}], [{"secondary_y": False}]]
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.50, 0.50],
        vertical_spacing=0.22,
        specs=specs,
        subplot_titles=[
            f"Aggregated L2 norm per layer — {ratio_name} | {method}",
            "Per-submodule breakdown — select a layer from the dropdown",
        ],
    )

    # Aggregated lines — primary y-axis
    fig.add_trace(go.Scatter(
        x=layers, y=before_vals,
        name="Before pruning",
        mode="lines+markers",
        line=dict(color=BEFORE_COLOR, dash="dash", width=1.5),
        marker=dict(size=5),
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=layers, y=after_vals,
        name=f"After ({method})",
        mode="lines+markers",
        line=dict(color=method_color, width=2),
        marker=dict(size=5),
    ), row=1, col=1, secondary_y=False)

    # Sparsity ratio — secondary y-axis (top panel only)
    if sparsity_ratios:
        ratio_layers = [l for l in layers if l in sparsity_ratios]
        ratio_vals   = [sparsity_ratios[l] for l in ratio_layers]
        fig.add_trace(go.Scatter(
            x=ratio_layers, y=ratio_vals,
            name="Sparsity ratio (rho_l)",
            mode="lines+markers",
            line=dict(color=RATIO_COLOR, dash="dot", width=1.5),
            marker=dict(size=4, symbol="diamond"),
            opacity=0.8,
        ), row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Sparsity ratio", secondary_y=True,
                         range=[0, 1.05], tickfont=dict(color=RATIO_COLOR),
                         title_font=dict(color=RATIO_COLOR), row=1, col=1)

    # Per-layer submodule bar traces — one before+after pair per layer.
    # All traces are pre-built; visibility is toggled by the dropdown.
    N_FIXED = 2 + (1 if sparsity_ratios else 0)  # always-visible traces above

    for i, layer in enumerate(layers):
        submodules = sorted(raw_before[layer].keys())
        b_vals     = [raw_before[layer].get(s, 0.0) for s in submodules]
        a_vals     = [raw_after[layer].get(s,  0.0) for s in submodules]
        visible    = (i == 0)

        fig.add_trace(go.Bar(
            x=submodules, y=b_vals,
            name="Before (submodules)",
            marker_color=BEFORE_COLOR, opacity=0.65,
            visible=visible, showlegend=False, offsetgroup=0,
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            x=submodules, y=a_vals,
            name=f"After ({method})",
            marker_color=method_color, opacity=0.85,
            visible=visible, showlegend=False, offsetgroup=1,
        ), row=2, col=1)

    n_layers = len(layers)

    def make_visibility(active_i):
        vis = [True] * N_FIXED
        for j in range(n_layers):
            vis.extend([j == active_i, j == active_i])
        return vis

    buttons = [
        dict(
            label=f"Layer {layer}",
            method="update",
            args=[
                {"visible": make_visibility(i)},
                {"annotations[1].text": f"Per-submodule breakdown — layer {layer}"},
            ],
        )
        for i, layer in enumerate(layers)
    ]

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.0, y=-0.06,
            xanchor="left", yanchor="top",
            buttons=buttons,
            active=0,
            showactive=True,
            bgcolor="white",
            bordercolor="#cccccc",
        )],
        barmode="group",
        height=700,
        template="plotly_white",
        legend=dict(orientation="h", x=0, y=1.06),
        margin=dict(b=120),
    )

    fig.update_xaxes(title_text="Layer index", row=1, col=1)
    fig.update_yaxes(title_text="L2 norm", row=1, col=1, secondary_y=False)
    fig.update_xaxes(title_text="Submodule", row=2, col=1, tickangle=-35)
    fig.update_yaxes(title_text="L2 norm", row=2, col=1)

    out = save_dir / f"weight_norm_interactive_{method}_{ratio_name}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Multi-method overlay: all methods for one ratio set
# (saved in {MODEL_NAME}_MDL subdir — spans all methods)
# ---------------------------------------------------------------------------

def plot_method_overlay(ratio_name, method_data, norm_type, ratio_dir):
    """
    method_data: {method: (before_dict, after_dict)}
    Top - Shows Absolute before (shared) and each method's after on the same axes.
          Saved in ratio_dir (the {MODEL_NAME}_MDL subdir) since it spans all methods.
    Bottom- Relative percentage drop (to highlight minute differences) 
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax_abs, ax_rel = axes

    first_method = next(iter(method_data))
    before, _ = method_data[first_method]
    layers = common_layers(before, *[a for _, a in method_data.values()])
    before_vals = [before[l] for l in layers]

    # --- Top Panel: Absolute Norms ---
    ax_abs.plot(layers, before_vals, color=BEFORE_COLOR, linewidth=1.8,
            linestyle="--", label="Before (shared)", zorder=2)
    
    for method, (_, after) in method_data.items():
        after_vals = [after[l] for l in layers]
        ax_abs.plot(layers, after_vals, color=METHOD_COLORS.get(method, "#333"),
                linewidth=2, label=f"After: {method}", zorder=3)

    ax_abs.set_ylabel("Absolute L2 Norm", fontsize=11, fontweight="bold")
    ax_abs.set_title(
        f"{norm_type.replace('_', ' ').title()} — Absolute Norms  |  {ratio_name}",
        fontsize=12, fontweight="bold",
    )
    ax_abs.legend(framealpha=0.9)
    ax_abs.grid(axis="y", alpha=0.3)

    # --- Bottom Panel: Relative Percentage Drop ---
    ax_rel.axhline(0, color=BEFORE_COLOR, linewidth=1.5, linestyle="--", label="Before (0% drop)", zorder=2)
    
    for method, (_, after) in method_data.items():
        drops = [100 * (before[l] - after[l]) / before[l] if before[l] != 0 else 0 for l in layers]
        ax_rel.plot(layers, drops, color=METHOD_COLORS.get(method, "#333"),
                linewidth=2, marker="o", markersize=4, 
                label=f"{method}", zorder=3)

    ax_rel.set_xlabel("Transformer layer index", fontsize=11)
    ax_rel.set_ylabel("Norm Drop (%)", fontsize=11, fontweight="bold")
    ax_rel.set_title(
        f"{norm_type.replace('_', ' ').title()} — Relative Drop (%)  |  {ratio_name}",
        fontsize=12, fontweight="bold",
    )
    ax_rel.legend(framealpha=0.9)
    ax_rel.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax_rel.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = ratio_dir / f"{norm_type}_method_overlay_{ratio_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Multi-ratio overlay: all ratio sets for one method
# (saved in output_dir — spans all ratio sets, no single natural home)
# ---------------------------------------------------------------------------

def plot_ratio_overlay(method, ratio_data, norm_type, output_dir, family):
    """
    Fix a pruning method and overlay all ratio sets on the same axes.

    ratio_data: {ratio_name: (before_dict, after_dict)}

    Answers: do different MDL ratio sets concentrate the norm drop in
    different layers, or do they produce the same profile shape?

    Saved in output_dir since it spans all ratio set directories.

    Top    — Absolute L2 norms (original design)
    Bottom — Relative percentage drop (to highlight minute differences)
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax_abs, ax_rel = axes

    first_ratio = next(iter(ratio_data))
    before, _ = ratio_data[first_ratio]
    all_afters = [a for _, a in ratio_data.values()]
    layers = common_layers(before, *all_afters)
    before_vals = [before[l] for l in layers]

    # --- Top Panel: Absolute Norms ---
    ax_abs.plot(layers, before_vals, color=BEFORE_COLOR, linewidth=1.8,
            linestyle="--", label="Before (shared baseline)", zorder=2)

    cmap = plt.cm.get_cmap("tab10", len(ratio_data))
    for i, (ratio_name, (_, after)) in enumerate(ratio_data.items()):
        after_vals = [after[l] for l in layers]
        # short_name = ratio_name[:35] + "…" if len(ratio_name) > 35 else ratio_name
        short_name = ratio_name.replace(f"_{family}", "")  # show only the unique part of the ratio name
        ax_abs.plot(layers, after_vals, color=cmap(i), linewidth=1.8,
                label=short_name, zorder=3)

    ax_abs.set_ylabel("Absolute L2 Norm", fontsize=11, fontweight="bold")
    ax_abs.set_title(
        f"{norm_type.replace('_', ' ').title()} — Absolute Norms  |  {method} |  {family}",
        fontsize=12, fontweight="bold",
    )
    ax_abs.legend(framealpha=0.9, fontsize=8, loc="upper right")
    ax_abs.grid(axis="y", alpha=0.3)

    # --- Bottom Panel: Relative Percentage Drop ---
    ax_rel.axhline(0, color=BEFORE_COLOR, linewidth=1.5, linestyle="--", label="Before (0% drop)", zorder=2)

    for i, (ratio_name, (_, after)) in enumerate(ratio_data.items()):
        drops = [100 * (before[l] - after[l]) / before[l] if before[l] != 0 else 0 for l in layers]
        # short_name = ratio_name[:35] + "…" if len(ratio_name) > 35 else ratio_name
        short_name = ratio_name.replace(f"_{family}", "")  # show only the unique part of the ratio name
        ax_rel.plot(layers, drops, color=cmap(i), linewidth=1.8, marker=".",
                label=short_name, zorder=3)

    ax_rel.set_xlabel("Transformer layer index", fontsize=11)
    ax_rel.set_ylabel("Norm Drop (%)", fontsize=11, fontweight="bold")
    ax_rel.set_title(
        f"{norm_type.replace('_', ' ').title()} — Relative Drop (%)  |  {method} |  {family}",
        fontsize=12, fontweight="bold",
    )
    ax_rel.legend(framealpha=0.9, fontsize=8, loc="upper right")
    ax_rel.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax_rel.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_dir / f"{norm_type}_ratio_overlay_{method}_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Summary heatmap across all ratio × method combos
# (saved in output_dir — spans everything)
# ---------------------------------------------------------------------------



def plot_delta_heatmap(delta_matrix, ratio_names, methods, norm_type, output_dir, family):
    """
    delta_matrix: shape (n_ratios, n_methods) — mean % norm drop
    Saved in output_dir since it summarises the entire experiment grid.
    """
    fig, ax = plt.subplots(figsize=(len(methods) * 2.2, len(ratio_names) * 0.9 + 1.2))

    vmax = max(abs(delta_matrix).max(), 1)
    im = ax.imshow(delta_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    datasets_name = [r.replace(f"_{family}", "") for r in ratio_names]

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_yticks(range(len(ratio_names)))
    ax.set_yticklabels(datasets_name, fontsize=9)
    ax.set_title(
        f"Mean {norm_type.replace('_', ' ')} drop (%) after pruning\n(green = bigger drop) {family}",
        fontsize=10, fontweight="bold",
    )

    for i in range(len(ratio_names)):
        for j in range(len(methods)):
            val = delta_matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="% drop")
    fig.tight_layout()

    out = output_dir / f"{norm_type}_delta_heatmap_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot weight / activation norms before and after pruning."
    )
    parser.add_argument("--experiment_dirs", type=str, nargs="+", required=True,
                        help=f"Explicit paths to the {MODEL_NAME}_MDL_* result directories to process.")
    parser.add_argument("--output_dir", type=str, default="plots/",
                        help="Directory for global plots that span all ratio sets.")
    parser.add_argument("--ratio_paths", type=str, nargs="*", default=None,
                        help=(
                            f"Explicit paths to the MDL ratio JSON files, one per ratio set, "
                            f"in the same alphabetical order as the {MODEL_NAME}_MDL_* result "
                            f"directories. When supplied, the sparsity ratio profile is "
                            f"overlaid on per-experiment norm plots. If omitted, ratio "
                            f"overlays are skipped."
                        ))
    parser.add_argument("--heat_map_repeat", action="store_true",
                        help=(
                            "If set, saves a copy of the heatmap in each family subdir "
                            "instead of just the global output_dir. This can be helpful "
                            "for browsing results by family without needing to open the "
                            "global summary directory."
                        ))
    parser.add_argument("--ratio_overlay_repeat", action="store_true",
                        help=(
                            "If set, saves a copy of each ratio overlay in each family subdir "
                            "instead of just the global output_dir. This can be helpful "
                            "for browsing results by family without needing to open the "
                            "global summary directory."
                        ))
    args = parser.parse_args()

    

    

    
    # Convert passed strings to Path objects, dropping any that don't actually exist
    ratio_dirs = []
    for d in args.experiment_dirs:
        p = Path(d)
        if p.exists() and p.is_dir():
            ratio_dirs.append(p)
        else:
            print(f"[warn] Provided experiment directory not found: {p}")

    if not ratio_dirs:
        print("No valid experiment directories provided. Exiting.")
        return

    # Extract the ratio names (e.g., "cola_mdl_minmax_0.55...")
    ratio_names = [d.name.removeprefix(f"{MODEL_NAME}_MDL_") for d in ratio_dirs]

    # # get the common suffix eg "mdl_minmax_0.55..."
    # common_suffix = get_common_suffix(ratio_names)
    # print(f"Identified common experiment suffix: {common_suffix}")

    # output_dir = Path(args.output_dir) / common_suffix
    # output_dir.mkdir(parents=True, exist_ok=True)
    # print(f"Global plots will be saved to: {output_dir}")





    # Create a dictionary linking the stripped ratio name exactly to its JSON path
    ratio_path_map = {}
    if args.ratio_paths:
        for path_str in args.ratio_paths:
            p = Path(path_str)
            
            # Use .name to get the full filename (including .55.json), 
            # then explicitly strip the .json extension to perfectly match the bash script
            clean_name = p.name.replace(".json", "")
            name_key = clean_name.removeprefix(f"rho_l_{MODEL_NAME}_")
            
            ratio_path_map[name_key] = path_str

    if args.ratio_paths and len(ratio_path_map) != len(ratio_dirs):
        print(f"[warn] Path count mismatch. Found {len(ratio_dirs)} valid directories "
              f"but mapped {len(ratio_path_map)} ratio JSONs. Proceeding safely.")
    
    # --- NEW: GROUP EXPERIMENTS BY FAMILY ---
    
    experiment_groups = defaultdict(list)
    for r_dir, r_name in zip(ratio_dirs, ratio_names):
        # r_name is like "cola_mdl_minmax_0.55[0:1]_0_3[1:4]_0_4" or "mrpc_mdl_minmax_0.95[0:1]_0_4"
        family = extract_family_suffix(r_name)
        # family eg mdl_minmax_0.55[0:1]_0_3[1:4]_0_4 or mdl_minmax_0.95[0:1]_0_4 etc
        experiment_groups[family].append((r_dir, r_name))
        
    print(f"\nFound {len(experiment_groups)} distinct experiment families.")

    for family, experiments in experiment_groups.items():

        print(f"\n{'='*60}")
        print(f"PROCESSING FAMILY: {family}")
        print(f"{'='*60}")

        ratio_group_dirs = [item[0] for item in experiments]
        ratio_group_names = [item[1] for item in experiments]

        # 1. Create the dynamic output directory for THIS family
        family_output_dir = Path(args.output_dir) / family
        family_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Global plots for this family -> {family_output_dir}")
          
        # Matrices for heatmaps: (n_ratios, n_methods)
        weight_delta_matrix     = np.full((len(ratio_group_dirs), len(METHODS)), np.nan)
        activation_delta_matrix = np.full((len(ratio_group_dirs), len(METHODS)), np.nan)

        # Accumulators for ratio overlay: {method: {ratio_name: (before, after)}}
        weight_ratio_data     = {m: {} for m in METHODS}
        activation_ratio_data = {m: {} for m in METHODS}

        for r_idx, (ratio_dir, ratio_name) in enumerate(zip(ratio_group_dirs, ratio_group_names)):
            print(f"\nProcessing ratio set: {ratio_name}")

            # Load sparsity ratios from the explicit path provided by the bash script
            exact_json_path = ratio_path_map.get(ratio_name)
            sparsity_ratios = load_ratio_path(exact_json_path)
            if sparsity_ratios:
                print(f"  loaded sparsity ratios ({len(sparsity_ratios)} layers) "
                    f"from {exact_json_path}")
            else:
                print("  [info] no ratio path provided — ratio overlay skipped for this set")

            weight_overlay_data     = {}
            activation_overlay_data = {}

            for m_idx, method in enumerate(METHODS):
                exp_dir = ratio_dir / method
                result  = load_experiment(exp_dir, ratio_name, method)
                if result is None:
                    print(f"  [{method}] missing norm files — skipping")
                    continue

                w_before, w_after, a_before, a_after, raw_w_before, raw_w_after = result
                print(f"  [{method}] loaded norms ({len(w_before)} layers)")

                # Per-experiment plots — saved in the method subdir (exp_dir)
                w_layers = common_layers(w_before, w_after)
                plot_before_after(w_layers, w_before, w_after, method, ratio_name,
                                "weight_norm", exp_dir,
                                sparsity_ratios=sparsity_ratios)

                a_layers = common_layers(a_before, a_after)
                plot_before_after(a_layers, a_before, a_after, method, ratio_name,
                                "activation_norm", exp_dir,
                                sparsity_ratios=sparsity_ratios)

                # Interactive plot — saved in the method subdir
                raw_layers = common_layers(w_before, w_after, raw_w_before, raw_w_after)
                plot_aggregated_weight_norms(
                    raw_layers, w_before, w_after,
                    raw_w_before, raw_w_after,
                    method, ratio_name, exp_dir,
                    sparsity_ratios=sparsity_ratios,
                )

                # Accumulate % drops for heatmap
                w_drops = [
                    100 * (w_before[l] - w_after[l]) / w_before[l]
                    for l in w_layers if w_before[l] != 0
                ]
                a_drops = [
                    100 * (a_before[l] - a_after[l]) / a_before[l]
                    for l in a_layers if a_before[l] != 0
                ]
                if w_drops:
                    weight_delta_matrix[r_idx, m_idx] = float(np.mean(w_drops))
                if a_drops:
                    activation_delta_matrix[r_idx, m_idx] = float(np.mean(a_drops))

                weight_overlay_data[method]     = (w_before, w_after)
                activation_overlay_data[method] = (a_before, a_after)

                weight_ratio_data[method][ratio_name]     = (w_before, w_after)
                activation_ratio_data[method][ratio_name] = (a_before, a_after)

            # Method overlay — saved in the {MODEL_NAME}_MDL subdir (ratio_dir)
            if weight_overlay_data:
                plot_method_overlay(ratio_name, weight_overlay_data,
                                    "weight_norm", ratio_dir)
            if activation_overlay_data:
                plot_method_overlay(ratio_name, activation_overlay_data,
                                    "activation_norm", ratio_dir)

        # Ratio overlay — saved in output_dir (spans all ratio sets)
        print("\nGenerating ratio overlay plots...")
        for method in METHODS:
            if args.ratio_overlay_repeat:
                method_out_dir = family_output_dir / method
                method_out_dir.mkdir(parents=True, exist_ok=True)
            else:
                method_out_dir = family_output_dir  # ratio overlay spans all methods, so save in family dir

            if weight_ratio_data[method]:
                plot_ratio_overlay(method, weight_ratio_data[method],
                                "weight_norm", method_out_dir, family)
            if activation_ratio_data[method]:
                plot_ratio_overlay(method, activation_ratio_data[method],
                                "activation_norm", method_out_dir, family)

        # Heatmaps — saved in output_dir (summarise entire experiment grid)
        short_ratio_names = [n[:40] for n in ratio_group_names]
        if args.heat_map_repeat:
            target_dirs = [family_output_dir] + ratio_group_dirs
        else:
            target_dirs = [family_output_dir]  # heatmap summarises all methods, so save in family dir only
        
        valid_w = ~np.isnan(weight_delta_matrix)
        if valid_w.any():
            for t_dir in target_dirs:
                plot_delta_heatmap(np.nan_to_num(weight_delta_matrix), short_ratio_names,
                                METHODS, "weight_norm", t_dir, family)

        valid_a = ~np.isnan(activation_delta_matrix)
        if valid_a.any():
            for t_dir in target_dirs:
                plot_delta_heatmap(np.nan_to_num(activation_delta_matrix), short_ratio_names,
                                METHODS, "activation_norm", t_dir, family)

        print(f"\nAll plots saved.")


if __name__ == "__main__":
    main()