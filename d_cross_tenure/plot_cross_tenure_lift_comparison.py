#!/usr/bin/env python
"""Replot existing cross-tenure results; no model training or scoring.
Bars: equal-weight mean of within-fold bucket lifts.
Spread: mean of within-fold highest/lowest lift ratios, NOT a ratio of mean bars.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go


TRANSFER = 'FROZEN_SOURCE_ON_TARGET'
NATIVE = 'TARGET_NATIVE_OUTER_OOF'
ROLES = (TRANSFER, NATIVE)
FILENAME = 'lift_by_outer_fold_equal_count_group.csv'
COLORS = [[0, '#2f5d8a'], [0.5, '#f2e6cf'], [1, '#a33b1f']]




def check(condition, message):
    if not condition:
        raise ValueError(message)




def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()




def fmt(value, pct=False):
    if not np.isfinite(value):
        return 'N/A'
    return f'{value:+.1f}%' if pct else f'{value:.3f}'




def strict_mean(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()) if np.isfinite(a).all() else np.nan




def load(folder, models, bins):
    path = folder / FILENAME
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. A spread-summary CSV cannot reconstruct all bucket bars.')
    df = pd.read_csv(path)
    required = {'fold', 'model_family', 'model_role', 'source_tenure', 'target_tenure',
                'n_bins', 'bin_rank', 'records', 'actual_events', 'lift'}
    check(required.issubset(df.columns), f'{path}: missing {sorted(required - set(df.columns))}')
    pair = df[['source_tenure', 'target_tenure']].drop_duplicates()
    check(len(pair) == 1 and pair.notna().all().all(), f'{path}: ambiguous tenure pair')
    source, target = [str(v).removesuffix('.0') for v in pair.iloc[0]]
    check(source != target, 'Use a cross-tenure output directory, not a native-baseline directory.')
    df.model_family = df.model_family.astype(str).str.lower().str.strip()
    df.model_role = df.model_role.astype(str).str.upper().str.strip()
    for name in ['fold', 'n_bins', 'bin_rank', 'records', 'actual_events']:
        v = pd.to_numeric(df[name], errors='raise')
        check(np.isfinite(v).all() and (v == np.floor(v)).all(), f'Invalid {name}: {path}')
        df[name] = v.astype('int64')
    df = df[df.model_family.isin(models) & df.n_bins.isin(bins)].copy()
    keys = ['model_family', 'model_role', 'fold', 'n_bins', 'bin_rank']
    check(not df.duplicated(keys).any(), f'Duplicate fold/bucket records: {path}')
    totals = {}
    for model in models:
        for n in bins:
            for role in ROLES:
                part = df[(df.model_family == model) & (df.n_bins == n) & (df.model_role == role)]
                check(set(part.fold) == {1, 2, 3, 4, 5}, f'{folder}: missing folds for {model}/{role}/{n} bins')
                for fold, g in part.groupby('fold'):
                    g = g.sort_values('bin_rank')
                    check(g.bin_rank.tolist() == list(range(1, n + 1)), 'Incomplete score-ranked buckets')
                    count, event = g.records.to_numpy(), g.actual_events.to_numpy()
                    check((count > 0).all() and (event >= 0).all() and (event <= count).all(), 'Invalid bucket counts')
                    check(count.max() - count.min() <= 1, 'Buckets are not equal-count')
                    total = (int(count.sum()), int(event.sum()))
                    check(total == totals.setdefault(fold, total), f'Mismatched target populations in fold {fold}')
                    rate = event.sum() / count.sum()
                    check(0 < rate < 1, 'An evaluation fold lacks both outcome classes')
                    lift = event / count / rate
                    check(np.allclose(lift, pd.to_numeric(g.lift), rtol=1e-7, atol=1e-10), 'Lift disagrees with bucket event counts')
                    df.loc[g.index, 'lift'] = lift
                    if {'minimum_score', 'maximum_score'}.issubset(g.columns):
                        lo, hi = g.minimum_score.to_numpy(float), g.maximum_score.to_numpy(float)
                        check(np.isfinite(lo).all() and np.isfinite(hi).all() and
                              (lo <= hi + 1e-12).all() and (lo[:-1] >= hi[1:] - 1e-12).all(), 'Buckets are not in descending predicted-score order')
    manifest_path = folder / 'evaluation_manifest.json'
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        check(manifest.get('evaluation_contract') == 'frozen_source_outer_fold_models_on_target_outer_oof', 'Wrong evaluation contract')
        check(manifest.get('final_holdout_rows_scored') == 0, 'Manifest does not confirm the final holdout is unscored')
        check(tuple(str(manifest.get(k)).removesuffix('.0') for k in ['source_tenure', 'target_tenure']) == (source, target), 'CSV/manifest tenure mismatch')
        check(manifest.get('output_artifacts', {}).get('lift_by_fold', {}).get('sha256') == digest(path), 'Lift CSV hash mismatch')
    else:
        print(f'WARNING: no evaluation_manifest.json in {folder}; provenance cannot be verified.')
    mp = folder / 'outer_oof_metrics_by_fold.csv'
    metrics = pd.read_csv(mp) if mp.is_file() else None
    if metrics is not None:
        required_metrics = {'fold', 'model_family', 'model_role', 'source_tenure', 'target_tenure', 'rows', 'top10_lift'}
        check(required_metrics.issubset(metrics.columns), f'{mp}: incomplete metric schema')
        check(set(metrics.source_tenure.astype(str).str.replace(r'\.0$', '', regex=True)) == {source} and
              set(metrics.target_tenure.astype(str).str.replace(r'\.0$', '', regex=True)) == {target}, 'Metric tenure identities disagree')
        if manifest is not None:
            check(manifest.get('output_artifacts', {}).get('metrics_by_fold', {}).get('sha256') == digest(mp), 'Metrics CSV hash mismatch')
    return dict(source=source, target=target, data=df, metrics=metrics, totals=totals, folder=folder)




def summarize(c, model, n):
    d = c['data']
    d = d[(d.model_family == model) & (d.n_bins == n)]
    # Keep score-ranked order. Do NOT sort by observed lift to force monotonic bars.
    means = d.groupby(['bin_rank', 'model_role']).lift.mean().unstack().reindex(range(n, 0, -1))
    s = dict(source_tenure=c['source'], target_tenure=c['target'], model_family=model, n_bins=n)
    fold_table = pd.DataFrame(index=pd.Index(range(1, 6), name='fold'))
    for role, name in [(TRANSFER, 'transfer'), (NATIVE, 'native')]:
        extremes = d[d.model_role == role].groupby('fold').lift.agg(['max', 'min'])
        fold_table[name + '_spread'] = extremes['max'] / extremes['min'].where(extremes['min'] > 0)
        fold_table[name + '_top10'] = np.nan
        if c['metrics'] is not None:
            m = c['metrics']
            m = m[(m.model_family == model) & (m.model_role == role)]
            check(not m.fold.duplicated().any() and set(m.fold) == {1, 2, 3, 4, 5}, 'Incomplete exact top10 metrics')
            m = m.set_index('fold').reindex(range(1, 6))
            check(np.array_equal(m.rows.to_numpy(), [c['totals'][f][0] for f in range(1, 6)]), 'Top10/lift row counts disagree')
            v = pd.to_numeric(m.top10_lift, errors='raise')
            check(np.isfinite(v).all() and (v >= 0).all(), 'Invalid exact top10 metric')
            fold_table[name + '_top10'] = v
    for metric in ['spread', 'top10']:
        t, b = fold_table['transfer_' + metric], fold_table['native_' + metric]
        tm, bm = strict_mean(t), strict_mean(b)
        s['transfer_' + metric], s['native_' + metric] = tm, bm
        s[metric + '_change_pct'] = 100 * (tm / bm - 1) if np.isfinite(tm) and np.isfinite(bm) and bm > 0 else np.nan
        fold_table[metric + '_delta'] = t - b
    return means, s, fold_table.reset_index()




def header(s):
    text = f"Mean fold spread (highest / lowest): transfer {fmt(s['transfer_spread'])} | native {fmt(s['native_spread'])} | change {fmt(s['spread_change_pct'], True)}"
    if np.isfinite(s['transfer_top10']):
        text += f"<br>Mean exact top-10% lift: transfer {fmt(s['transfer_top10'])} | native {fmt(s['native_top10'])} | change {fmt(s['top10_change_pct'], True)}"
    return text




def plot(c, model, tables, bins, maximum):
    fig = go.Figure()
    initial = 20 if 20 in bins else max(bins)
    buttons = []
    def annotations(n):
        return [dict(x=0, y=1.24, xref='paper', yref='paper', xanchor='left', yanchor='bottom',
                     showarrow=False, align='left', text=header(tables[n][1]), font=dict(size=14)),
                dict(x=0, y=-0.29, xref='paper', yref='paper', xanchor='left', showarrow=False, align='left',
                     text='Bars/line: equal-weight mean of within-fold lifts. Spread: mean of fold ratios, not ratio of displayed bars.<br>Same target population; each model forms its own ranked bins. Transfer evidence, not a pooled-model test.', font=dict(size=11))]
    for n in bins:
        means = tables[n][0]
        x = [f'{r * 100 / n:g}' for r in means.index]
        t, b = means[TRANSFER].to_numpy(), means[NATIVE].to_numpy()
        fig.add_trace(go.Bar(x=x, y=t, text=[f'{v:.2f}' for v in t], textposition='outside', cliponaxis=False,
                            name=f"Transfer {c['source']} â†’ {c['target']}", visible=n == initial,
                            marker=dict(color=t, coloraxis='coloraxis', line=dict(width=0)),
                            customdata=np.column_stack([b, t - b]),
                            hovertemplate='Percentile: %{x}%<br>Transfer lift: %{y:.4f}<br>Native lift: %{customdata[0]:.4f}<br>Transfer minus native: %{customdata[1]:+.4f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=x, y=b, mode='lines+markers', name=f"Native {c['target']} â†’ {c['target']}",
                                visible=n == initial, line=dict(color='#222222', width=2), marker=dict(size=6),
                                hovertemplate='Percentile: %{x}%<br>Native lift: %{y:.4f}<extra></extra>'))
        buttons.append(dict(label=str(n), method='update', args=[{'visible': [k == n for k in bins for _ in ROLES]},
                           {'xaxis.categoryarray': x, 'xaxis.range': [-0.6, n - 0.4], 'annotations': annotations(n)}]))
    fig.update_layout(template='plotly_white', height=880, margin=dict(l=75, r=45, t=215, b=175),
                      title=dict(text=f"{model.upper()} | Target tenure {c['target']} policies", x=0.045, y=0.94, font=dict(size=22)),
                      font=dict(family='Arial', size=13), bargap=0.12,
                      xaxis=dict(title='Percentile (low predicted churn risk on left; high on right)', type='category', categoryorder='array',
                                 categoryarray=[f'{r * 100 / initial:g}' for r in range(initial, 0, -1)], range=[-0.6, initial - 0.4]),
                      yaxis=dict(title='Churn lift', range=[0, maximum * 1.13], gridcolor='rgba(0,0,0,0.15)', griddash='dot'),
                      updatemenus=[dict(type='buttons', direction='right', x=0, y=1.12, active=bins.index(initial), buttons=buttons)],
                      coloraxis=dict(cmin=0, cmax=maximum, colorscale=COLORS,
                                     colorbar=dict(title='Lift', orientation='h', x=0.67, y=1.065, xanchor='center', yanchor='bottom', len=0.34, thickness=13)),
                      legend=dict(orientation='h', x=0, y=-0.18), annotations=annotations(initial))
    fig.add_hline(y=1, line_dash='dash', line_color='gray', line_width=1.3)
    return fig




def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison-dirs', nargs='+', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--models', nargs='+', choices=['xgboost', 'logistic'], default=['xgboost', 'logistic'])
    p.add_argument('--bins', nargs='+', type=int, default=[3, 4, 5, 10, 20])
    a = p.parse_args()
    bins = sorted(set(a.bins))
    check(min(bins) >= 2, 'Need at least two bins')
    comparisons = [load(d, a.models, bins) for d in a.comparison_dirs]
    pairs = [(c['source'], c['target']) for c in comparisons]
    check(len(pairs) == len(set(pairs)), 'Duplicate source-target pairs supplied')
    jobs, summary, folds, buckets = [], [], [], []
    maximum = 1.0
    for c in comparisons:
        for model in a.models:
            tables = {n: summarize(c, model, n) for n in bins}
            for n, (means, s, f) in tables.items():
                maximum = max(maximum, float(means[list(ROLES)].to_numpy().max()))
                summary.append(s)
                identity = {k: s[k] for k in ['source_tenure', 'target_tenure', 'model_family', 'n_bins']}
                folds.append(f.assign(**identity))
                buckets.append(means.reset_index().assign(**identity))
            jobs.append((c, model, tables))
    out = a.out_dir.resolve()
    check(not out.exists(), f'Output already exists: {out}. Use a fresh output directory.')
    out.mkdir(parents=True)
    links = []
    for c, model, tables in jobs:
        name = f"{model}_s{c['source']}_to_t{c['target']}_vs_native".replace('+', 'plus')
        fig = plot(c, model, tables, bins, maximum)
        fig.write_html(out / (name + '.html'), include_plotlyjs=True, config={'displaylogo': False, 'responsive': True,
                       'toImageButtonOptions': {'format': 'png', 'filename': name, 'width': 1800, 'height': 1000, 'scale': 2}})
        links.append(f'<p><a href="{name}.html">{html.escape(name)}</a></p>')
        print(f"{name}: {header(tables[20 if 20 in bins else max(bins)][1]).replace('<br>', '; ')}")
    pd.DataFrame(summary).to_csv(out / 'comparison_statistics.csv', index=False)
    pd.concat(folds, ignore_index=True).to_csv(out / 'fold_comparison_statistics.csv', index=False)
    pd.concat(buckets, ignore_index=True).to_csv(out / 'bin_lift_comparison.csv', index=False)
    (out / 'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Cross-tenure lift</title></head><body style="font:18px Arial;margin:40px"><h1>Cross-tenure lift comparisons</h1><p>Colored bars = transfer. Black line = target-native. Use the chart camera button for PNG export.</p>' + ''.join(links) + '</body></html>', encoding='utf-8')
    (out / 'plot_manifest.json').write_text(json.dumps({'inputs': [{'directory': str(c['folder'].resolve()), 'lift_csv_sha256': digest(c['folder'] / FILENAME)} for c in comparisons], 'aggregation': 'equal-weight mean fold lifts; mean within-fold highest/lowest spread', 'script_sha256': digest(Path(__file__)), 'model_training_or_scoring': False, 'row_identity_verification': 'Inherited from upstream evaluator; aggregate CSVs cannot independently re-prove row identity'}, indent=2), encoding='utf-8')
    print(f'\nOpen: {out / "index.html"}\nNo model training, rescoring, or source-output changes.')




if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
