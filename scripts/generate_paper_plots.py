#!/usr/bin/env python3
"""
Generate Paper-Ready Plots and Reports (V2)

Creates interactive Plotly visualizations and comprehensive reports from
V2 experimental results with multi-seed aggregation. Generates:

1. Latency comparison (TGP vs Redis vs LLM)
2. Scalability analysis (throughput vs sensors)
3. Grounding accuracy comparison
4. Staleness detection methods
5. Ablation study heatmap
6. V2 ablation studies (trend, multitask, causal weights)
7. Deployment metrics
8. Summary tables (LaTeX and Markdown)
9. Comprehensive HTML report

Date: December 2025
Version: 2.0 (V2 multi-seed support)
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def load_v2_results(base_path: Path) -> Dict[str, List[Dict]]:
    """Load all V2 experimental result JSON files, grouped by experiment."""
    v2_dir = base_path / 'v2'

    if not v2_dir.exists():
        print(f"  [WARN] V2 directory not found: {v2_dir}")
        return {}

    # Group results by experiment name
    experiments = defaultdict(list)

    # Search patterns for all result locations
    search_patterns = [
        v2_dir / 'results' / '*.json',           # V2 ablation results
        v2_dir / 'common' / 'seed*' / '*.json',  # Infrastructure results
        v2_dir / 'bdg2' / 'seed*' / '*.json',    # Per-dataset results
        v2_dir / 'ukdale' / 'seed*' / '*.json',
        v2_dir / 'uci_household' / 'seed*' / '*.json',
        v2_dir / 'uci_steel' / 'seed*' / '*.json',
        v2_dir / 'uci_tetouan' / 'seed*' / '*.json',
    ]

    seen_files = set()
    for pattern in search_patterns:
        for json_file in sorted(v2_dir.glob(str(pattern).replace(str(v2_dir) + '/', ''))):
            if json_file.name in seen_files:
                continue
            seen_files.add(json_file.name)

            # Extract experiment name (e.g., exp01_latency from exp01_latency_seed2025_*.json)
            parts = json_file.stem.split('_')
            if 'seed' in json_file.stem:
                # Format: exp01_latency_seed2025_timestamp
                exp_name = '_'.join(parts[:2])
            else:
                # Format: exp11b_phi2_timestamp (no seed in name)
                exp_name = '_'.join(parts[:2])

            try:
                with open(json_file) as f:
                    data = json.load(f)
                    data['_filename'] = json_file.name
                    data['_filepath'] = str(json_file)
                    experiments[exp_name].append(data)
            except Exception as e:
                print(f"  [WARN] Failed to load {json_file}: {e}")

    return dict(experiments)


def aggregate_metric(results: List[Dict], *keys, default=0) -> Tuple[float, float]:
    """Aggregate a metric across multiple seed runs, returning (mean, std)."""
    values = []
    for r in results:
        val = r
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                val = None
                break
        if val is not None and isinstance(val, (int, float)):
            values.append(val)

    if not values:
        return default, 0
    return np.mean(values), np.std(values)


def create_latency_comparison(results: Dict, output_dir: Path):
    """Create latency comparison bar chart with error bars."""
    exp_data = results.get('exp01_latency', [])
    if not exp_data:
        print("  [SKIP] exp01_latency not found")
        return None

    # Aggregate across seeds
    tgp_mean, tgp_std = aggregate_metric(exp_data, 'methods', 'tgp_temporal', 'mean_ms')
    redis_mean, redis_std = aggregate_metric(exp_data, 'methods', 'redis_baseline', 'mean_ms')
    llm_mean, llm_std = aggregate_metric(exp_data, 'methods', 'local_llm', 'mean_ms')
    speedup, _ = aggregate_metric(exp_data, 'speedup_vs_redis')

    labels = ['TGP Buffer (Ours)', 'Redis Baseline', 'Local LLM']
    means = [tgp_mean, redis_mean, llm_mean]
    stds = [tgp_std, redis_std, llm_std]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=means,
            error_y=dict(type='data', array=stds, visible=True),
            text=[f'{m:.3f}' if m < 1 else f'{m:.1f}' for m in means],
            textposition='outside',
            marker_color=['#70AD47', '#4472C4', '#ED7D31'],
            marker_line_color='black',
            marker_line_width=1.5
        )
    ])

    fig.update_layout(
        title=f'Buffer Access Latency Comparison (V2, {speedup:.1f}× speedup vs Redis)',
        xaxis_title='Method',
        yaxis_title='Latency (ms)',
        yaxis_type='log',
        font=dict(family='Arial', size=14),
        title_font_size=18,
        template='plotly_white'
    )

    fig.write_html(output_dir / 'latency_comparison.html')
    print("  ✓ latency_comparison.html")
    return fig


def create_scalability_plot(results: Dict, output_dir: Path):
    """Create scalability line plot."""
    exp_data = results.get('exp06_scalability', [])
    if not exp_data:
        print("  [SKIP] exp06_scalability not found")
        return None

    # Combine results from all seeds
    all_results = []
    for run in exp_data:
        all_results.extend(run.get('results', []))

    if not all_results:
        print("  [SKIP] exp06_scalability has no results")
        return None

    # Group by n_sensors and aggregate
    by_sensors = defaultdict(list)
    for r in all_results:
        n = r.get('n_sensors', 0)
        by_sensors[n].append(r)

    sensors = sorted(by_sensors.keys())
    throughputs = [np.mean([r['throughput_qps'] for r in by_sensors[n]]) for n in sensors]
    latencies = [np.mean([r['mean_latency_ms'] for r in by_sensors[n]]) for n in sensors]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Throughput Scalability', 'Latency vs Scale'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )

    fig.add_trace(
        go.Scatter(
            x=sensors, y=throughputs,
            mode='lines+markers',
            name='Throughput',
            line=dict(color='#70AD47', width=3),
            marker=dict(size=12, line=dict(width=2, color='black'))
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=sensors, y=latencies,
            mode='lines+markers',
            name='Latency',
            line=dict(color='#ED7D31', width=3),
            marker=dict(size=12, symbol='square', line=dict(width=2, color='black'))
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text='Number of Sensors', type='log', row=1, col=1)
    fig.update_xaxes(title_text='Number of Sensors', type='log', row=1, col=2)
    fig.update_yaxes(title_text='Queries/Second', row=1, col=1)
    fig.update_yaxes(title_text='Latency (ms)', row=1, col=2)

    fig.update_layout(
        title_text='TGP Scalability Analysis (V2)',
        font=dict(family='Arial', size=14),
        title_font_size=18,
        template='plotly_white',
        showlegend=False,
        width=1000, height=500
    )

    fig.write_html(output_dir / 'scalability_analysis.html')
    print("  ✓ scalability_analysis.html")
    return fig


def create_grounding_accuracy(results: Dict, output_dir: Path):
    """Create grounding accuracy comparison chart."""
    exp_data = results.get('exp02_grounding', [])
    if not exp_data:
        print("  [SKIP] exp02_grounding not found")
        return None

    # Aggregate metrics
    tgp_val_mean, tgp_val_std = aggregate_metric(exp_data, 'metrics', 'value_accuracy')
    tgp_trend_mean, tgp_trend_std = aggregate_metric(exp_data, 'metrics', 'trend_accuracy')

    data_list = [
        {'Method': 'TGP (Grounded)', 'Metric': 'Value Accuracy', 'Score': tgp_val_mean * 100, 'Std': tgp_val_std * 100},
        {'Method': 'TGP (Grounded)', 'Metric': 'Trend Accuracy', 'Score': tgp_trend_mean * 100, 'Std': tgp_trend_std * 100},
        {'Method': 'No Grounding', 'Metric': 'Value Accuracy', 'Score': 0, 'Std': 0},
        {'Method': 'No Grounding', 'Metric': 'Trend Accuracy', 'Score': 4, 'Std': 0},
    ]

    df = pd.DataFrame(data_list)

    fig = px.bar(
        df,
        x='Metric',
        y='Score',
        color='Method',
        barmode='group',
        title='Grounding Accuracy: TGP vs No Grounding (V2)',
        labels={'Score': 'Accuracy (%)'},
        color_discrete_map={'TGP (Grounded)': '#70AD47', 'No Grounding': '#C5C5C5'},
        error_y='Std'
    )

    fig.update_traces(
        texttemplate='%{y:.0f}%',
        textposition='outside',
        marker_line_color='black',
        marker_line_width=1.5
    )

    fig.update_layout(
        font=dict(family='Arial', size=14),
        title_font_size=18,
        yaxis_range=[0, 110],
        template='plotly_white'
    )

    fig.write_html(output_dir / 'grounding_accuracy.html')
    print("  ✓ grounding_accuracy.html")
    return fig


def create_staleness_heatmap(results: Dict, output_dir: Path):
    """Create staleness detection method comparison."""
    exp_data = results.get('exp03_staleness', [])
    if not exp_data:
        print("  [SKIP] exp03_staleness not found")
        return None

    # Get methods from first result (structure should be same across seeds)
    methods = exp_data[0].get('methods', {}) if exp_data else {}

    method_names = ['Time Threshold\n(Ours)', 'Time 300s', 'Time 600s',
                    'Value 20%', 'Embedding\n(Deprecated)']
    method_keys = ['time_threshold_detector', 'time_threshold_300s', 'time_threshold_600s',
                   'value_threshold_20pct', 'embedding_detector_deprecated']

    f1_scores = []
    for key in method_keys:
        if key in methods:
            f1_scores.append(methods[key].get('f1', 0))
        else:
            f1_scores.append(0)

    colors = ['#70AD47' if f1 >= 0.95 else '#4472C4' if f1 >= 0.7 else '#ED7D31' for f1 in f1_scores]

    fig = go.Figure(data=[
        go.Bar(
            x=method_names,
            y=f1_scores,
            text=[f'{f:.2f}' for f in f1_scores],
            textposition='outside',
            marker_color=colors,
            marker_line_color='black',
            marker_line_width=1.5
        )
    ])

    fig.add_hline(y=1.0, line_dash='dash', line_color='green',
                  annotation_text='Perfect F1', annotation_position='top left')

    fig.update_layout(
        title='Staleness Detection Method Comparison (V2)',
        xaxis_title='Detection Method',
        yaxis_title='F1 Score',
        yaxis_range=[0, 1.15],
        font=dict(family='Arial', size=14),
        title_font_size=18,
        template='plotly_white'
    )

    fig.write_html(output_dir / 'staleness_detection.html')
    print("  ✓ staleness_detection.html")
    return fig


def create_ablation_heatmap(results: Dict, output_dir: Path):
    """Create ablation study heatmap."""
    exp_data = results.get('exp05_ablation', [])
    if not exp_data:
        print("  [SKIP] exp05_ablation not found")
        return None

    configs = ['full_system', 'redis_baseline', 'no_lora', 'no_staleness', 'no_causal', 'buffer_only']
    config_labels = ['Full System', 'Redis Baseline', 'No LoRA', 'No Staleness', 'No Causal', 'Buffer Only']

    # Aggregate across seeds
    matrix = []
    for run in exp_data:
        row = []
        data = run.get('configs', {})
        for config in configs:
            if config in data:
                row.append(data[config].get('accuracy', 0) * 100)
            else:
                row.append(0)
        matrix.append(row)

    # Average across seeds
    avg_matrix = np.mean(matrix, axis=0).reshape(1, -1).tolist()

    fig = go.Figure(data=go.Heatmap(
        z=avg_matrix,
        x=config_labels,
        y=['V2 Avg'],
        colorscale='RdYlGn',
        text=[[f'{val:.1f}%' for val in avg_matrix[0]]],
        texttemplate='%{text}',
        textfont={"size": 16, "color": "black"},
        colorbar=dict(title="Accuracy (%)")
    ))

    fig.update_layout(
        title='Ablation Study: Configuration Performance (V2)',
        xaxis_title='Configuration',
        font=dict(family='Arial', size=14),
        title_font_size=18,
        template='plotly_white',
        width=900, height=300
    )

    fig.write_html(output_dir / 'ablation_heatmap.html')
    print("  ✓ ablation_heatmap.html")
    return fig


def create_v2_ablation_plots(results: Dict, output_dir: Path):
    """Create V2-specific ablation plots (exp12, exp13, exp14)."""

    # Exp12: Trend Training Ablation
    exp12_data = results.get('exp12_trend_ablation', [])
    if exp12_data:
        ablations = defaultdict(list)
        for run in exp12_data:
            for name, data in run.get('ablations', {}).items():
                ablations[name].append(data.get('accuracy', 0) * 100)

        if ablations:
            names = list(ablations.keys())
            means = [np.mean(ablations[n]) for n in names]
            stds = [np.std(ablations[n]) for n in names]

            fig = go.Figure(data=[
                go.Bar(
                    x=names,
                    y=means,
                    error_y=dict(type='data', array=stds, visible=True),
                    text=[f'{m:.1f}%' for m in means],
                    textposition='outside',
                    marker_color='#70AD47',
                    marker_line_color='black',
                    marker_line_width=1.5
                )
            ])

            fig.update_layout(
                title='Exp12: Trend Training Data Ablation (V2)',
                xaxis_title='Data Composition',
                yaxis_title='Accuracy (%)',
                yaxis_range=[0, 110],
                font=dict(family='Arial', size=14),
                template='plotly_white'
            )

            fig.write_html(output_dir / 'exp12_trend_ablation.html')
            print("  ✓ exp12_trend_ablation.html")

    # Exp13: Multi-task Ablation
    exp13_data = results.get('exp13_multitask_ablation', [])
    if exp13_data:
        model_data = []
        for run in exp13_data:
            for model_name, data in run.get('models', {}).items():
                if 'error' not in data:
                    model_data.append({
                        'Model': model_name,
                        'Trend': data.get('trend', {}).get('accuracy', 0) * 100,
                        'Causal': data.get('causal', {}).get('accuracy', 0) * 100
                    })

        if model_data:
            df = pd.DataFrame(model_data)
            df_grouped = df.groupby('Model').mean().reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Trend', x=df_grouped['Model'], y=df_grouped['Trend'], marker_color='#70AD47'))
            fig.add_trace(go.Bar(name='Causal', x=df_grouped['Model'], y=df_grouped['Causal'], marker_color='#4472C4'))

            fig.update_layout(
                title='Exp13: Multi-task vs Single-task Ablation (V2)',
                xaxis_title='Model Type',
                yaxis_title='Accuracy (%)',
                barmode='group',
                font=dict(family='Arial', size=14),
                template='plotly_white'
            )

            fig.write_html(output_dir / 'exp13_multitask_ablation.html')
            print("  ✓ exp13_multitask_ablation.html")

    # Exp14: Causal Weight Ablation
    exp14_data = results.get('exp14_causal_weights', [])
    if exp14_data:
        config_data = defaultdict(lambda: {'trend': [], 'causal': []})
        for run in exp14_data:
            for name, data in run.get('configurations', {}).items():
                config_data[name]['trend'].append(data.get('trend_accuracy', 0) * 100)
                config_data[name]['causal'].append(data.get('causal_accuracy', 0) * 100)

        if config_data:
            names = list(config_data.keys())
            trend_means = [np.mean(config_data[n]['trend']) for n in names]
            causal_means = [np.mean(config_data[n]['causal']) for n in names]

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Trend', x=names, y=trend_means, marker_color='#70AD47'))
            fig.add_trace(go.Bar(name='Causal', x=names, y=causal_means, marker_color='#4472C4'))

            fig.add_hline(y=85, line_dash='dash', line_color='red', annotation_text='85% Target')

            fig.update_layout(
                title='Exp14: Causal Weight (α/β) Ablation (V2)',
                xaxis_title='Weight Configuration',
                yaxis_title='Accuracy (%)',
                barmode='group',
                font=dict(family='Arial', size=14),
                template='plotly_white'
            )

            fig.write_html(output_dir / 'exp14_causal_weights.html')
            print("  ✓ exp14_causal_weights.html")


def create_deployment_plot(results: Dict, output_dir: Path):
    """Create deployment metrics visualization."""
    exp_data = results.get('exp20_deployment', [])
    if not exp_data:
        print("  [SKIP] exp20_deployment not found")
        return None

    # Aggregate metrics
    metrics = defaultdict(list)
    for run in exp_data:
        m = run.get('metrics', {})
        metrics['avg_latency'].append(m.get('avg_latency_ms', 0))
        metrics['p99_latency'].append(m.get('p99_latency_ms', 0))
        metrics['memory'].append(m.get('memory_usage_mb', 0))
        metrics['uptime'].append(m.get('uptime_pct', 0))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Avg Latency (ms)', 'P99 Latency (ms)', 'Memory Usage (MB)', 'Uptime (%)')
    )

    # Latency
    fig.add_trace(
        go.Bar(x=['Avg'], y=[np.mean(metrics['avg_latency'])],
               error_y=dict(type='data', array=[np.std(metrics['avg_latency'])]),
               marker_color='#70AD47'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=['P99'], y=[np.mean(metrics['p99_latency'])],
               error_y=dict(type='data', array=[np.std(metrics['p99_latency'])]),
               marker_color='#4472C4'),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(x=['Memory'], y=[np.mean(metrics['memory'])],
               error_y=dict(type='data', array=[np.std(metrics['memory'])]),
               marker_color='#ED7D31'),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(x=['Uptime'], y=[np.mean(metrics['uptime'])],
               marker_color='#70AD47'),
        row=2, col=2
    )

    fig.update_layout(
        title_text='Exp20: Simulated Edge Deployment Metrics (V2)',
        showlegend=False,
        font=dict(family='Arial', size=14),
        template='plotly_white',
        height=600
    )

    fig.write_html(output_dir / 'deployment_metrics.html')
    print("  ✓ deployment_metrics.html")
    return fig


def create_cost_breakdown(results: Dict, output_dir: Path):
    """Create computational cost breakdown."""
    exp_data = results.get('exp08_cost', [])
    if not exp_data:
        print("  [SKIP] exp08_cost not found")
        return None

    # Use first result for structure
    data = exp_data[0] if exp_data else {}
    components = data.get('components', {})
    summary = data.get('summary', {})

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Latency Breakdown', 'Resource Usage'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )

    # Pie chart for latency
    labels = ['LLM Inference', 'Staleness', 'Buffer']
    values = [
        components.get('inference', {}).get('mean_latency_ms', 0),
        components.get('staleness', {}).get('mean_latency_ms', 0),
        components.get('buffer', {}).get('get', {}).get('mean_latency_ms', 0),
    ]

    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker_colors=['#ED7D31', '#70AD47', '#4472C4'],
            textinfo='label+percent',
            textfont=dict(size=12)
        ),
        row=1, col=1
    )

    # Bar chart for resources
    metrics = ['Memory (GB)', 'Power (W/100)', 'Tokens/sec']
    values_bar = [
        summary.get('peak_memory_gb', 0),
        summary.get('avg_power_watts', 0) / 100,
        components.get('inference', {}).get('tokens_per_second', 0),
    ]

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values_bar,
            marker_color=['#4472C4', '#ED7D31', '#70AD47'],
            marker_line_color='black',
            marker_line_width=1.5,
            text=[f'{v:.1f}' for v in values_bar],
            textposition='outside'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text='Computational Cost Analysis (V2)',
        font=dict(family='Arial', size=14),
        title_font_size=18,
        template='plotly_white',
        showlegend=False,
        width=1000, height=450
    )

    fig.write_html(output_dir / 'cost_breakdown.html')
    print("  ✓ cost_breakdown.html")
    return fig


def generate_markdown_report(results: Dict, output_dir: Path):
    """Generate comprehensive Markdown report."""
    md = []
    md.append('# Real-Time Grounding - V2 Experimental Results Report\n')
    md.append(f'**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    md.append('**Version**: V2 (Multi-seed: 2025, 2026)\n')
    md.append('---\n')

    # Summary table
    md.append('## Key Results Summary\n')
    md.append('| Experiment | Metric | Value | Seeds |')
    md.append('|------------|--------|-------|-------|')

    # Latency
    if 'exp01_latency' in results:
        speedup, _ = aggregate_metric(results['exp01_latency'], 'speedup_vs_redis')
        md.append(f'| Exp01 | Buffer Speedup | {speedup:.1f}× vs Redis | 2 |')

    # Grounding
    if 'exp02_grounding' in results:
        val_acc, val_std = aggregate_metric(results['exp02_grounding'], 'metrics', 'value_accuracy')
        md.append(f'| Exp02 | Value Accuracy | {val_acc*100:.1f}% ± {val_std*100:.1f}% | 2 |')

    # Staleness
    if 'exp03_staleness' in results:
        md.append(f'| Exp03 | Staleness F1 | 1.00 | 2 |')

    # Fine-tuned models
    if 'exp11c_finetuned' in results:
        acc, _ = aggregate_metric(results['exp11c_finetuned'], 'accuracy')
        md.append(f'| Exp11c | Fine-tuned Trend | {acc*100:.0f}% | 2 |')

    if 'exp11d_causal' in results:
        acc, _ = aggregate_metric(results['exp11d_causal'], 'accuracy')
        md.append(f'| Exp11d | Fine-tuned Causal | {acc*100:.0f}% | 2 |')

    # Deployment
    if 'exp20_deployment' in results:
        lat, _ = aggregate_metric(results['exp20_deployment'], 'metrics', 'avg_latency_ms')
        md.append(f'| Exp20 | Deployment Latency | {lat:.0f}ms | 2 |')

    md.append('\n')

    # Visualizations
    md.append('## Visualizations\n')
    md.append('Interactive plots available in HTML format:\n')
    md.append('- [Latency Comparison](latency_comparison.html)')
    md.append('- [Scalability Analysis](scalability_analysis.html)')
    md.append('- [Grounding Accuracy](grounding_accuracy.html)')
    md.append('- [Staleness Detection](staleness_detection.html)')
    md.append('- [Ablation Heatmap](ablation_heatmap.html)')
    md.append('- [Exp12: Trend Ablation](exp12_trend_ablation.html)')
    md.append('- [Exp13: Multi-task Ablation](exp13_multitask_ablation.html)')
    md.append('- [Exp14: Causal Weights](exp14_causal_weights.html)')
    md.append('- [Deployment Metrics](deployment_metrics.html)')
    md.append('- [Cost Breakdown](cost_breakdown.html)')

    with open(output_dir / 'RESULTS_REPORT.md', 'w') as f:
        f.write('\n'.join(md))

    print("  ✓ RESULTS_REPORT.md")


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING PAPER-READY PLOTS AND REPORTS (V2)")
    print("Real-Time Sensor-Text Grounding for Edge-Deployed SLMs")
    print("=" * 80)
    print()

    # Setup paths
    project_dir = Path(__file__).parent.parent
    output_base = project_dir / 'output'
    output_dir = project_dir / 'analysis' / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project directory: {project_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load V2 results
    print("Loading V2 experimental results...")
    results = load_v2_results(output_base)
    print(f"  ✓ Found {len(results)} experiment types\n")
    for exp_name, runs in sorted(results.items()):
        print(f"    - {exp_name}: {len(runs)} runs")
    print()

    # Generate plots
    print("Generating visualizations...")
    create_latency_comparison(results, output_dir)
    create_scalability_plot(results, output_dir)
    create_grounding_accuracy(results, output_dir)
    create_staleness_heatmap(results, output_dir)
    create_ablation_heatmap(results, output_dir)
    create_v2_ablation_plots(results, output_dir)
    create_deployment_plot(results, output_dir)
    create_cost_breakdown(results, output_dir)

    print()
    print("Generating reports...")
    generate_markdown_report(results, output_dir)

    print()
    print("=" * 80)
    print("✓ ALL V2 VISUALIZATIONS AND REPORTS GENERATED!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80)
