"""
Builds index.html (the post) from the result files in data/.

    python build_post.py

Inputs (all produced by the scripts in code/nanoGPT/, see README.md):
    data/v22_dynamics.json, v23_dynamics.json, v33_dynamics.json   probe_combiner_dynamics.py
    data/v22_carry_probes.json                                     probe_combiner_carry.py
    data/v22_attention.json                                        probe_combiner_attn.py
    data/v22_patching.json                                         probe_combiner_patch.py
    data/v22_scratch_tokens.json                                   probe_thinking_tokens.py
    data/v22_eval.json, v23_eval.json, v33_eval.json               eval_additive.py
    data/v33_limits_r80.json, v33_limits_r140.json                 eval_additive_limits.py
    data/logs/                                                     the raw run logs (a few numbers are read from them)
    data/base_style.css                                            shared page style (same as the other two posts)
Charts are inline SVG (no scripts); colours are CSS variables, so they follow light/dark mode.
"""
import html
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(HERE, 'data', *p)

# ---------------------------------------------------------------------------------------------------------
# chart helpers (same as the Rope-aliasing post)

W, H = 680, 300
ML, MR, MT, MB = 52, 128, 14, 44          # margins; the right margin holds direct labels


def _x(v, lo, hi):
    return ML + (v - lo) / (hi - lo) * (W - ML - MR)


def _y(v, lo, hi):
    return MT + (1 - (v - lo) / (hi - lo)) * (H - MT - MB)


def line_chart(series, xdom, ydom, xticks, yticks, xlabel, ylabel, aria, shade=(), vline=None, ytickfmt=None,
               xtickfmt=None, hline=None):
    """series: dicts with name, xs, ys, var (css colour variable), dash ('' / '6 4' / '2 3'), width,
    optional lo/hi (min-max band), label (direct label text, at the last point)."""
    x0, x1 = xdom
    y0, y1 = ydom
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(aria)}">']
    for a, b, txt in shade:
        xa, xb = _x(max(a, x0), x0, x1), _x(min(b, x1), x0, x1)
        out.append(f'<rect x="{xa:.1f}" y="{MT}" width="{xb - xa:.1f}" height="{H - MT - MB}" class="shade"/>')
        if txt:
            out.append(f'<text x="{xa + 4:.1f}" y="{MT + 12}" class="tick shade-label">{html.escape(txt)}</text>')
    for t in yticks:
        y = _y(t, y0, y1)
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{(ytickfmt or str)(t)}</text>')
    for t in xticks:
        x = _x(t, x0, x1)
        out.append(f'<text x="{x:.1f}" y="{H - MB + 18}" class="tick" text-anchor="middle">{(xtickfmt or str)(t)}</text>')
    out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{H - MB}" y2="{H - MB}" class="axis"/>')
    out.append(f'<text x="{(ML + W - MR) / 2:.1f}" y="{H - 6}" class="axis-label" text-anchor="middle">{html.escape(xlabel)}</text>')
    out.append(f'<text x="14" y="{(MT + H - MB) / 2:.1f}" class="axis-label" text-anchor="middle" '
               f'transform="rotate(-90 14 {(MT + H - MB) / 2:.1f})">{html.escape(ylabel)}</text>')
    if hline is not None:
        y = _y(hline, y0, y1)
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="ref"/>')
    if vline is not None:
        x = _x(vline, x0, x1)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{MT}" y2="{H - MB}" class="ref"/>')
    for s in series:
        if 'lo' in s:
            pts = [(_x(a, x0, x1), _y(b, y0, y1)) for a, b in zip(s['xs'], s['hi'])]
            pts += [(_x(a, x0, x1), _y(b, y0, y1)) for a, b in reversed(list(zip(s['xs'], s['lo'])))]
            out.append(f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" '
                       f'style="fill: var({s["var"]}); opacity: 0.12"/>')
    labels = []
    for s in series:
        pts = [(_x(a, x0, x1), _y(min(max(b, y0), y1), y0, y1)) for a, b in zip(s['xs'], s['ys'])]
        d = 'M' + ' L'.join(f'{a:.1f},{b:.1f}' for a, b in pts)
        dash = f'; stroke-dasharray: {s["dash"]}' if s.get('dash') else ''
        out.append(f'<path d="{d}" class="series" style="stroke: var({s["var"]}); stroke-width: {s.get("width", 2)}{dash}"/>')
        if s.get('markers'):
            for (a, b), xv, yv in zip(pts, s['xs'], s['ys']):
                out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" class="dot" style="fill: var({s["var"]})"/>')
        yfmt = ytickfmt or (lambda v: f'{v:.2f}')
        xfmt = xtickfmt or (lambda v: f'{v:g}')
        if len(pts) <= 40:                                    # hover targets with native tooltips
            for (a, b), xv, yv in zip(pts, s['xs'], s['ys']):
                out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="7" class="hit"><title>'
                           f'{html.escape(s["name"])}: {xfmt(xv)} &#8594; {yfmt(yv)}</title></circle>')
        if s.get('label'):
            labels.append([pts[-1][1], s['label'], s['var'], s.get('dash', '')])
    labels.sort()
    for i in range(1, len(labels)):                       # nudge direct labels apart (downwards) ...
        if labels[i][0] - labels[i - 1][0] < 14:
            labels[i][0] = labels[i - 1][0] + 14
    floor = H - MB - 2                                    # ... then shift the stack up if it ran past the axis
    if labels and labels[-1][0] > floor:
        labels[-1][0] = floor
        for i in range(len(labels) - 2, -1, -1):
            labels[i][0] = min(labels[i][0], labels[i + 1][0] - 14)
    for y, txt, var, dash in labels:                      # a short swatch in the series colour, then the label in neutral ink
        dd = f'; stroke-dasharray: {dash}' if dash else ''
        out.append(f'<line x1="{W - MR + 4}" x2="{W - MR + 18}" y1="{y:.1f}" y2="{y:.1f}" '
                   f'style="stroke: var({var}); stroke-width: 2.6{dd}"/>')
        out.append(f'<text x="{W - MR + 23}" y="{y + 4:.1f}" class="direct">{html.escape(txt)}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def table(caption, head, rows, cls=None):
    th = ''.join(f'<th>{h}</th>' for h in head)
    body = ''
    for r in rows:
        trcls = ''
        if isinstance(r, dict):
            trcls, r = f' class="{r["cls"]}"', r['cells']
        body += f'<tr{trcls}>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>'
    return (f'<div class="table-wrap"><table{f" class=\"{cls}\"" if cls else ""}><caption>{caption}</caption>'
            f'<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')


def pct(v):
    return f'{round(100 * v)}%'


def p1(v):
    """percent with one decimal, but 100% / 0% stay clean"""
    if v >= 0.9995:
        return '100%'
    if v <= 0.0005:
        return '0%'
    return f'{100 * v:.1f}%'


def cell(v, good=0.99, bad=0.5, fmt=p1):
    cls = 'hi' if v >= good else ('lo' if v <= bad else '')
    return f'<span class="{cls}">{fmt(v)}</span>' if cls else fmt(v)


def tdcell(v, good=0.99, bad=0.5, fmt=p1):
    """a table cell string whose <td> carries the hi/lo class (td.hi / td.lo are in base_style.css)"""
    return f'{fmt(v)}'


# ---------------------------------------------------------------------------------------------------------
# data

def load(name):
    return json.load(open(D(name), encoding='utf-8'))


def log(name):
    return open(D('logs', name), encoding='utf-8').read()


MODELS = ('v22', 'v23', 'v33')
dyn = {m: load(f'{m}_dynamics.json') for m in MODELS}
ev = {m: load(f'{m}_eval.json') for m in MODELS}
carry = load('v22_carry_probes.json')
attn = load('v22_attention.json')
patch = load('v22_patching.json')
scr = load('v22_scratch_tokens.json')
lim = {80: load('v33_limits_r80.json'), 140: load('v33_limits_r140.json')}

TIERS = ['test_indist', 'test_gap', 'test_hint', 'test_ood', 'test_far_ood']
TLAB = {'test_indist': '1–10 digits', 'test_gap': '11–13', 'test_hint': '14–18',
        'test_ood': '19–21', 'test_far_ood': '24–26'}
BUDGET = {m: dyn[m]['repeat_max'] for m in MODELS}           # trained round budget: 24 / 24 / 80
EXTRA = 40                                                    # the probe runs every model this many rounds past its budget
for m in MODELS:
    assert dyn[m]['extended_rounds'] == BUDGET[m] + EXTRA, m
assert (dyn['v22']['iter'], dyn['v23']['iter'], dyn['v33']['iter']) == (9500, 10500, 15000)
assert BUDGET == {'v22': 24, 'v23': 24, 'v33': 80}
TRAIN_MAX = {'v22': 20, 'v23': 20, 'v33': 64}                 # composed_train_max_digits in the three configs
for m in MODELS:
    cfg = open(os.path.join(HERE, 'code', 'nanoGPT', 'config', f'train_addition_additive_{m}.py'), encoding='utf-8').read()
    got = int(re.search(r'^composed_train_max_digits\s*=\s*(\d+)', cfg, re.M).group(1))
    assert got == TRAIN_MAX[m], (m, got)

# the logs say which checkpoint each result file came from
assert 'ckpt_name = ckpt_15000.pt' in log('v33_eval.log')
assert 'ckpt_name = ckpt_15000.pt' in log('v33_limits_r80.log') or lim[80]['ckpt'].endswith('ckpt_15000.pt')
assert lim[80]['ckpt'].endswith('ckpt_15000.pt') and lim[140]['ckpt'].endswith('ckpt_15000.pt')


def curve(m, tier):
    return dyn[m]['A3_acc_by_tier'][tier]['curve']


def sample_n(m):                                              # problems per tier in the round-by-round probe (printed in the run log)
    mm = re.search(r'test_indist\s+n=\s*(\d+)', log(f'{m}_probe_dynamics.log'))
    return int(mm.group(1))


NS = {m: sample_n(m) for m in MODELS}
assert NS == {'v22': 150, 'v23': 150, 'v33': 60}, NS


def acc(m, tier):
    return ev[m][tier]['sum_accuracy']


# round at which accuracy first reaches 99% on the sample, and the lowest accuracy afterwards (to the last round run)
def settle(m, tier):
    c = curve(m, tier)
    first = next((i for i, v in enumerate(c) if v >= 0.99), None)
    return first, (min(c[first:]) if first is not None else None)


# chain statistics from the crafted carry chains (A2)
def chain_stats(m):
    org, inter, term, last_is_end, last_is_term = [], [], [], 0, 0
    chains = dyn[m]['A2_chains']
    for c in chains:
        j, mm, fc = c['j'], c['m'], c['first_correct_by_place']
        o, t = fc[str(j)], fc[str(j + mm)]
        i_ = [fc[str(p)] for p in range(j + 1, j + mm)]
        if o is not None:
            org.append(o)
        if t is not None:
            term.append(t)
        inter += [v for v in i_ if v is not None]
        vals = {p: fc[str(p)] for p in range(j, j + mm + 1) if fc[str(p)] is not None}
        if vals and max(vals.values()) in (o, t):
            last_is_end += 1
        if vals and t is not None and t == max(vals.values()):
            last_is_term += 1
    mean = lambda xs: sum(xs) / len(xs)
    return dict(org=mean(org), inter=mean(inter), term=mean(term), n=len(chains), last_is_end=last_is_end, last_is_term=last_is_term,
                org_rng=(min(org), max(org)), inter_rng=(min(inter), max(inter)), term_rng=(min(term), max(term)))


CS = {m: chain_stats(m) for m in MODELS}
print('last-to-settle counts (v22): ends', CS['v22']['last_is_end'], 'terminus', CS['v22']['last_is_term'], 'of', CS['v22']['n'])

# the v22 table above is the original GPU evaluation; the CPU re-run in the logs must agree with it to 4 decimals
for _m in MODELS:
    _cpu = [float(x) for x in re.findall(r'sum accuracy:\s+([\d.]+)', log(f'{_m}_eval.log'))]
    assert _cpu == [round(ev[_m][t]['sum_accuracy'], 4) for t in TIERS], (_m, _cpu)
assert all(settle('v33', t)[1] == 1.0 for t in TIERS), 'v33 is not flat on every length set'
assert all(settle('v23', t)[1] == 1.0 for t in ('test_indist', 'test_gap', 'test_hint')), 'v23 is not flat on its short sets'

# longest carry chain in the probe problems (A4): mean_conv_round - mean_slack
MEAN_CHAIN = {m: dyn[m]['A4_convergence']['mean_conv_round'] - dyn[m]['A4_convergence']['mean_slack_vs_chainlen'] for m in MODELS}

# E: drift
def E(m, fam):
    v = dyn[m]['E_update_norms'][fam]
    rp = v['rel_update_place']
    return dict(r12=rp[11], r24=rp[23], r48=rp[47], last=rp[-1], nlast=len(rp), cos=v['late_update_cos'])


# F4 slope over places >= 3
F4 = patch['F4_natural_wavefront']
_ks = [int(k) for k in F4['first_flip_by_k'] if int(k) >= 3]
_ys = [F4['first_flip_by_k'][str(k)] for k in _ks]
_mx, _my = sum(_ks) / len(_ks), sum(_ys) / len(_ys)
F4_SLOPE_HI = sum((x - _mx) * (y - _my) for x, y in zip(_ks, _ys)) / sum((x - _mx) ** 2 for x in _ks)

# scratch-token probe baselines are printed in the run log, not stored in the json
_tl = log('v22_probe_scratch_tokens.log')
B3_BASE = {k: float(re.search(rf'{k}\s+baseline=([\d.]+)', _tl).group(1)) for k in ('total_carry', 'digit_count', 'max_chain')}

# the donor-graft run log (rounds 12-16: the grafted place flips, nothing above it does)
_pl = log('v22_probe_patch.log')
for r in (12, 14, 16):
    assert re.search(rf'inject@r{r}: full-ripple 0\.00\s+first-flip by k=\[{r}\.0, None, None, None, None, None, None\]', _pl), r

# ---------------------------------------------------------------------------------------------------------
# charts

TVAR = {'test_indist': '--t1', 'test_gap': '--t2', 'test_hint': '--t3', 'test_ood': '--t4', 'test_far_ood': '--t5'}


def acc_chart(m):
    xmax = BUDGET[m] + EXTRA
    series = []
    for t in TIERS:
        c = curve(m, t)
        assert len(c) == xmax + 1
        series.append(dict(name=TLAB[t], xs=list(range(xmax + 1)), ys=c, var=TVAR[t], label=TLAB[t], width=2))
    step = 8 if xmax <= 64 else 20
    return line_chart(series, (0, xmax), (0, 1.0), list(range(0, xmax + 1, step)), [0, .25, .5, .75, 1.0],
                      'rounds of the combiner block', 'exact-match accuracy',
                      f'Accuracy against number of combiner rounds for {m}, by problem length',
                      shade=[(BUDGET[m], xmax, f'past the trained {BUDGET[m]} rounds')], vline=BUDGET[m],
                      ytickfmt=lambda v: f'{round(100 * v)}%', xtickfmt=lambda v: f'{v:g}')


CHART_ACC = {m: acc_chart(m) for m in MODELS}

# C: what a place token knows, by round
PP = carry['place_probe']
rr = list(range(25))
chart_probe = line_chart(
    [dict(name='local digit sum', xs=rr, ys=PP['sum_nc'], var='--s3', label='local digit sum'),
     dict(name='carry out', xs=rr, ys=PP['carry_out'], var='--s2', label='carry out'),
     dict(name='carry in', xs=rr, ys=PP['carry_in'], var='--s4', label='carry in'),
     dict(name='answer digit', xs=rr, ys=PP['final'], var='--s1', label='answer digit', width=2.6)],
    (0, 24), (0.5, 1.0), [0, 4, 8, 12, 16, 20, 24], [.5, .6, .7, .8, .9, 1.0],
    'rounds of the combiner block (0 = before the first)', 'linear probe, held-out accuracy',
    'What a place token linearly encodes, by round', ytickfmt=lambda v: f'{round(100 * v)}%')

PT = carry['place_probe_by_type']
TYPES = [('chain_origin', 'carry starts here', '--s3'), ('chain_interior', 'inside a carry', '--s4'),
         ('chain_terminus', 'carry ends here', '--s2'), ('no_carry', 'no carry', '--ctx')]
chart_type_final = line_chart(
    [dict(name=n, xs=rr, ys=PT['final'][k], var=v, label=n) for k, n, v in TYPES],
    (0, 24), (0, 1.0), [0, 4, 8, 12, 16, 20, 24], [0, .25, .5, .75, 1.0],
    'rounds of the combiner block', 'answer digit decodable',
    'Answer-digit probe accuracy by kind of place', ytickfmt=lambda v: f'{round(100 * v)}%')
chart_type_cin = line_chart(
    [dict(name=n, xs=rr, ys=PT['carry_in'][k], var=v, label=n) for k, n, v in TYPES],
    (0, 24), (0, 1.0), [0, 4, 8, 12, 16, 20, 24], [0, .25, .5, .75, 1.0],
    'rounds of the combiner block', 'carry-in decodable',
    'Carry-in probe accuracy by kind of place', ytickfmt=lambda v: f'{round(100 * v)}%')

# D: attention offsets (the json arrays are indexed from 0 = the first application of the block; the post counts rounds from 1)
AB = attn['buckets']
BINS = AB['rand16']['offset_bins']
sel = [i for i, b in enumerate(BINS) if -8 <= b <= 8]
off_x = [BINS[i] for i in sel]
for b in AB.values():
    pk = BINS[max(range(len(b['offset_profile_round6'])), key=lambda i: b['offset_profile_round6'][i])]
    assert pk == -1
chart_off = line_chart(
    [dict(name='random 16-digit, round 7', xs=off_x, ys=[AB['rand16']['offset_profile_round6'][i] for i in sel], var='--s1',
          label='random, round 7', markers=True),
     dict(name='run of nines, 16 digits, round 7', xs=off_x, ys=[AB['nines16']['offset_profile_round6'][i] for i in sel], var='--s2',
          label='nines, round 7', markers=True),
     dict(name='run of nines, 16 digits, round 17', xs=off_x, ys=[AB['nines16']['offset_profile_round16'][i] for i in sel], var='--s2',
          label='nines, round 17', dash='6 4')],
    (-8, 8), (0, 0.21), [-8, -6, -4, -2, 0, 2, 4, 6, 8], [0, .05, .10, .15, .20],
    'place of the attended token relative to the attending token (−1 = the next place down)',
    'share of attention', 'Attention by relative place in the combiner block', ytickfmt=lambda v: f'{round(100 * v)}%',
    xtickfmt=lambda v: f'{v:+d}' if v else '0', vline=0)

R24 = list(range(1, 25))
chart_scr = line_chart(
    [dict(name='random 6-digit', xs=R24, ys=AB['rand6']['place->think_by_round'], var='--ctx', label='6 digits'),
     dict(name='random 16-digit', xs=R24, ys=AB['rand16']['place->think_by_round'], var='--s1', label='16 digits'),
     dict(name='run of nines, 20-digit', xs=R24, ys=AB['nines20']['place->think_by_round'], var='--s2', label='20 digits, nines')],
    (1, 24), (0, 0.5), [1, 4, 8, 12, 16, 20, 24], [0, .1, .2, .3, .4, .5],
    'round (1 = the first application of the block)', 'share of a place token’s attention',
    'Share of place-token attention that goes to the four scratch tokens, by round', ytickfmt=lambda v: f'{round(100 * v)}%')

# F: patching
F1 = patch['F1_injection']
f1_rounds = sorted(int(k) for k in F1 if k.isdigit())
chart_f1 = line_chart(
    [dict(name='the grafted place’s answer digit goes up by one', xs=f1_rounds, ys=[F1[str(r)]['flip_rate'] for r in f1_rounds],
          var='--s1', label='digit +1', markers=True, width=2.4),
     dict(name='any other place changes', xs=f1_rounds, ys=[1 - F1[str(r)]['clean_elsewhere'] for r in f1_rounds],
          var='--ctx', label='others change', dash='4 3')],
    (1, 14), (0, 1.0), [1, 2, 4, 6, 8, 10, 12, 14], [0, .25, .5, .75, 1.0],
    'round at which the state was swapped in', 'fraction of 240 problems', 'Grafting a carried-in state into one place',
    ytickfmt=lambda v: f'{round(100 * v)}%')

ff = F4['first_flip_by_k']
f4k = sorted(int(k) for k in ff)
chart_f4 = line_chart(
    [dict(name='first round the digit is right', xs=f4k, ys=[ff[str(k)] for k in f4k], var='--s1', markers=True, width=2.2,
          label='measured'),
     dict(name='fit', xs=[0, 11], ys=[F4['intercept'], F4['intercept'] + F4['slope_rounds_per_place'] * 11], var='--s2', dash='6 4',
          label=f'{F4["slope_rounds_per_place"]:.2f} rounds/place')],
    (0, 11), (0, 26), [0, 2, 4, 6, 8, 10], [0, 5, 10, 15, 20, 24],
    'place along a run of nines (0 = where the carry enters)', 'round the digit becomes 0',
    'When a carry reaches each place of a run of nines')

# limits
LD = sorted(int(k) for k in lim[80]['natural'])
chart_lim = line_chart(
    [dict(name='natural, 80 rounds', xs=LD, ys=[lim[80]['natural'][str(d)] for d in LD], var='--s1', markers=True, label='random'),
     dict(name='carry-heavy, 80 rounds', xs=LD, ys=[lim[80]['carry_heavy'][str(d)] for d in LD], var='--s2', markers=True, label='carry-heavy'),
     dict(name='natural, 140 rounds', xs=LD, ys=[lim[140]['natural'][str(d)] for d in LD], var='--s1', dash='2 3', width=3, label='random, 140'),
     dict(name='carry-heavy, 140 rounds', xs=LD, ys=[lim[140]['carry_heavy'][str(d)] for d in LD], var='--s2', dash='2 3', width=3,
          label='carry-heavy, 140')],
    (24, 124), (0, 1.0), [24, 48, 64, 80, 96, 112, 124], [0, .25, .5, .75, 1.0],
    'digits in the longer operand', 'exact-match accuracy', 'v33 accuracy against problem length, 80 and 140 rounds',
    shade=[(64, 126, 'past the longest training problem')], vline=64, ytickfmt=lambda v: f'{round(100 * v)}%')

# pipeline diagram
DIAGRAM = '''
<svg viewBox="0 0 870 282" role="img" aria-label="Pipeline: two numbers are encoded digit by digit into place slots, a transformer block is applied repeatedly to place tokens and four scratch tokens, and each place is decoded to a digit in parallel">
  <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill: var(--ink-faint)"/></marker></defs>
  <g style="font: 12.5px 'Newsreader', serif; fill: var(--ink)">
    <text x="12" y="30" style="font: 11px 'JetBrains Mono', monospace; fill: var(--ink-faint)">INPUT</text>
    <g style="font: 600 15px 'JetBrains Mono', monospace">
      <text x="12" y="62">1 2 3 4 5</text><text x="12" y="92">+ 6 7 8</text>
    </g>
    <text x="12" y="124" style="fill: var(--ink-soft)">two numbers,</text><text x="12" y="141" style="fill: var(--ink-soft)">digit by digit</text>

    <line x1="112" y1="76" x2="146" y2="76" style="stroke: var(--ink-faint)" stroke-width="1.5" marker-end="url(#arr)"/>

    <rect x="150" y="14" width="210" height="210" rx="9" style="fill: var(--surface-2); stroke: var(--line)"/>
    <text x="165" y="38" style="font: 11px 'JetBrains Mono', monospace; fill: var(--accent); letter-spacing: .08em">1 &#183; ENCODE</text>
    <text x="165" y="62" style="font: 600 14px 'Fraunces', serif">fixed codec</text>
    <text x="165" y="88" style="fill: var(--ink-soft)">each digit becomes 3 numbers</text>
    <text x="165" y="106" style="fill: var(--ink-soft)">(one of 10 learned codewords)</text>
    <text x="165" y="132" style="fill: var(--ink-soft)">written into that place&#8217;s own</text>
    <text x="165" y="150" style="fill: var(--ink-soft)">slot, units first. 30 slots;</text>
    <text x="165" y="168" style="fill: var(--ink-soft)">short numbers padded with 0s</text>
    <text x="165" y="206" style="font: 11px 'JetBrains Mono', monospace; fill: var(--ink-faint)">~40 learned numbers</text>

    <line x1="362" y1="119" x2="396" y2="119" style="stroke: var(--ink-faint)" stroke-width="1.5" marker-end="url(#arr)"/>

    <rect x="400" y="14" width="270" height="210" rx="9" style="fill: var(--surface); stroke: var(--accent); stroke-width: 1.5"/>
    <text x="415" y="38" style="font: 11px 'JetBrains Mono', monospace; fill: var(--accent); letter-spacing: .08em">2 &#183; COMBINE</text>
    <text x="662" y="30" text-anchor="end" style="font: 11px 'JetBrains Mono', monospace; fill: var(--accent)">&#215; N rounds</text>
    <text x="415" y="60" style="font: 600 14px 'Fraunces', serif">the learned part</text>
    <g style="font: 10.5px 'JetBrains Mono', monospace">
      <rect x="415" y="72" width="24" height="20" rx="3" style="fill: var(--surface-2); stroke: var(--line)"/><text x="421" y="86" style="fill: var(--ink-soft)">p0</text>
      <rect x="443" y="72" width="24" height="20" rx="3" style="fill: var(--surface-2); stroke: var(--line)"/><text x="449" y="86" style="fill: var(--ink-soft)">p1</text>
      <rect x="471" y="72" width="24" height="20" rx="3" style="fill: var(--surface-2); stroke: var(--line)"/><text x="477" y="86" style="fill: var(--ink-soft)">p2</text>
      <text x="500" y="86" style="fill: var(--ink-faint)">&#8230;</text>
      <rect x="516" y="72" width="30" height="20" rx="3" style="fill: var(--surface-2); stroke: var(--line)"/><text x="520" y="86" style="fill: var(--ink-soft)">p29</text>
      <rect x="560" y="72" width="18" height="20" rx="3" style="fill: var(--accent); opacity: .85"/>
      <rect x="582" y="72" width="18" height="20" rx="3" style="fill: var(--accent); opacity: .85"/>
      <rect x="604" y="72" width="18" height="20" rx="3" style="fill: var(--accent); opacity: .85"/>
      <rect x="626" y="72" width="18" height="20" rx="3" style="fill: var(--accent); opacity: .85"/>
    </g>
    <path d="M 635 96 C 635 124, 426 124, 426 96" fill="none" style="stroke: var(--accent)" stroke-width="1.5" marker-end="url(#arr)"/>
    <text x="415" y="148" style="fill: var(--ink-soft)">30 place tokens + 4 scratch tokens</text>
    <text x="415" y="168" style="fill: var(--ink-soft)">one attention + MLP block, the same</text>
    <text x="415" y="185" style="fill: var(--ink-soft)">weights applied again and again</text>
    <text x="415" y="202" style="fill: var(--ink-soft)">(24 rounds; 80 in v33)</text>
    <text x="415" y="216" style="font: 11px 'JetBrains Mono', monospace; fill: var(--ink-faint)">1.8M parameters, almost all here</text>

    <line x1="672" y1="119" x2="706" y2="119" style="stroke: var(--ink-faint)" stroke-width="1.5" marker-end="url(#arr)"/>

    <rect x="710" y="14" width="150" height="210" rx="9" style="fill: var(--surface-2); stroke: var(--line)"/>
    <text x="724" y="38" style="font: 11px 'JetBrains Mono', monospace; fill: var(--accent); letter-spacing: .08em">3 &#183; DECODE</text>
    <text x="724" y="62" style="font: 600 14px 'Fraunces', serif">no parameters</text>
    <text x="724" y="88" style="fill: var(--ink-soft)">each place becomes</text>
    <text x="724" y="106" style="fill: var(--ink-soft)">the nearest of the</text>
    <text x="724" y="124" style="fill: var(--ink-soft)">10 codewords, all</text>
    <text x="724" y="142" style="fill: var(--ink-soft)">places at once</text>
    <g style="font: 600 15px 'JetBrains Mono', monospace"><text x="724" y="200">1 3 0 2 3</text></g>
  </g>
  <text x="12" y="262" style="font: 12.5px 'Newsreader', serif; fill: var(--ink-soft)">Everything before and after the loop is fixed or nearly so. All the arithmetic, including every carry, happens in the loop.</text>
</svg>
'''

EXTRA_CSS = '''
  :root { --s1: #00849c; --s2: #c4501c; --s3: #7a55b0; --s4: #3d8a3a; --ctx: #8a8d92;
          --t1: #a9d3d9; --t2: #6bb6c0; --t3: #2f98a6; --t4: #0c6571; --t5: #c4501c; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
          --s1: #1fa3ae; --s2: #d9703e; --s3: #a98bd8; --s4: #6fbf6a; --ctx: #7d7f85;
          --t1: #355a60; --t2: #4a8a93; --t3: #5fb1bb; --t4: #94dde6; --t5: #e08a63; } }
  :root[data-theme="dark"] { --s1: #1fa3ae; --s2: #d9703e; --s3: #a98bd8; --s4: #6fbf6a; --ctx: #7d7f85;
          --t1: #355a60; --t2: #4a8a93; --t3: #5fb1bb; --t4: #94dde6; --t5: #e08a63; }
  svg.chart { width: 100%; height: auto; display: block; margin: 0.4rem 0 1.4rem; overflow: visible; }
  svg.chart .grid { stroke: var(--line-soft); stroke-width: 1; }
  svg.chart .axis { stroke: var(--line); stroke-width: 1; }
  svg.chart .ref { stroke: var(--ink-faint); stroke-width: 1; stroke-dasharray: 3 3; }
  svg.chart .shade { fill: var(--ink-faint); opacity: 0.10; }
  svg.chart .tick { fill: var(--ink-faint); font: 11px 'JetBrains Mono', monospace; }
  svg.chart .shade-label { fill: var(--ink-soft); paint-order: stroke; stroke: var(--bg); stroke-width: 4px; stroke-linejoin: round; }
  .table-wrap table.compact th, .table-wrap table.compact td { padding: 0.6rem 0.65rem; font-size: 13px; }
  .table-wrap table.compact { min-width: 0; }
  svg.chart .axis-label { fill: var(--ink-soft); font: 12px 'Newsreader', serif; }
  svg.chart .direct { fill: var(--ink-soft); font: 12px 'Newsreader', serif; }
  svg.chart .series { fill: none; stroke-linejoin: round; stroke-linecap: round; }
  svg.chart .dot { stroke: var(--surface); stroke-width: 2; }
  svg.chart .hit { fill: transparent; }
  .fig-cap { font-size: 0.88rem; color: var(--ink-soft); margin: -0.6rem 0 1.6rem; }
  .fig-title { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; letter-spacing: 0.06em; text-transform: uppercase;
               color: var(--ink-faint); margin: 1.8rem 0 0; }
  ol.steps > li, ul.points > li { margin-bottom: 0.7rem; }
  td .hi { color: var(--ok); font-weight: 600; }
  td .lo { color: var(--fail); font-weight: 600; }
  .refs li { font-size: 0.92rem; margin-bottom: 0.35rem; }
'''

base_css = open(D('base_style.css'), encoding='utf-8').read()

# ---------------------------------------------------------------------------------------------------------
# numbers used in the prose and tables

v22, v23, v33 = 'v22', 'v23', 'v33'
DUEL_L = curve(v22, 'test_ood')[BUDGET[v22] + EXTRA]
DUEL_R = curve(v33, 'test_ood')[BUDGET[v33] + EXTRA]
DUEL_L0 = curve(v22, 'test_ood')[BUDGET[v22]]
A4 = {m: dyn[m]['A4_convergence'] for m in MODELS}
A2 = {m: dyn[m]['A2_settle_slope']['slope_rounds_per_place'] for m in MODELS}
P_R = lambda k, r: PP[k][r]
PT_R = lambda k, t, r: PT[k][t][r]
base = carry['baseline']
SIDE = AB['rand16']
ABN = AB['nines16']
th_rand16, th_nines20 = AB['rand16']['place->think_by_round'], AB['nines20']['place->think_by_round']
B1 = scr['B1_ablation']
B3 = scr['B3_probe']
R16 = {int(j): v for j, v in scr['B2_attn_role']['rand16'].items()}
N20 = {int(j): v for j, v in scr['B2_attn_role']['nines20'].items()}
# claims in the text about which scratch token is read most
assert all(max(R16[0]['received_from_place'][:10]) > max(R16[j]['received_from_place'][:10]) for j in (1, 2, 3))
assert all(N20[3]['received_from_place'][i] > max(N20[j]['received_from_place'][i] for j in (0, 1, 2)) for i in range(3, 9))
assert B1['zero_all']['test_indist'] == B1['mean_all']['test_indist'] == 0.0
assert all(B1[f'{k}_0'][c] == 0.0 for k in ('zero', 'mean') for c in B1['baseline'])
assert all(B1['zero_2'][c] >= B1['baseline'][c] - 0.01 for c in B1['baseline'])
assert all(PT_ok > 0.99 for PT_ok in (carry['place_probe_by_type']['final'][k][16] for k in ('chain_origin', 'chain_interior', 'chain_terminus', 'no_carry')))
for r in (8, 10):
    assert re.search(rf'inject@r{r}: full-ripple 0\.00\s+first-flip by k=\[None, None, None, None, None, None, None\]', log('v22_probe_patch.log')), r
far_by_digit = {m: ev[m]['test_far_ood']['sum_by_digits'] for m in MODELS}
LN, LC = lim[80]['natural'], lim[80]['carry_heavy']
LN2, LC2 = lim[140]['natural'], lim[140]['carry_heavy']
max_diff_140 = max(abs(LN[k] - LN2[k]) for k in LN) , max(abs(LC[k] - LC2[k]) for k in LC)
assert max(max_diff_140) <= 0.02 + 1e-9, max_diff_140

# ---------------------------------------------------------------------------------------------------------
# tables

def t_tiers():
    rows = []
    for t in TIERS:
        rows.append([f'{TLAB[t]} <span style="opacity:.6">(n={ev["v22"][t]["n"]})</span>'] + [cell(acc(m, t), 0.995, 0.5) for m in MODELS])
    rows.append({'cls': 'split', 'cells': ['longest training problem'] + [f'{TRAIN_MAX[m]} digits' for m in MODELS]})
    rows.append(['rounds of the block'] + [f'{BUDGET[m]}' for m in MODELS])
    return table('Exact-match accuracy on the fixed test sets', ['problem length', 'v22', 'v23', 'v33'], rows)


def t_far():
    rows = []
    for d in ('24', '25', '26'):
        rows.append([f'{d} digits'] + [cell(far_by_digit[m][d], 0.995, 0.5) for m in MODELS])
    return table('The 24–26 digit set, by length (200 problems each)', ['', 'v22', 'v23', 'v33'], rows)


def t_timing():
    rows = []
    rows.append(['mean round at which the answer stops changing'] + [f'{A4[m]["mean_conv_round"]:.1f}' for m in MODELS])
    rows.append(['correlation with the number of digits'] + [f'{A4[m]["corr_convround_vs_digits"]:.2f}' for m in MODELS])
    rows.append(['correlation with the longest carry chain'] + [f'{A4[m]["corr_convround_vs_chainlen"]:.2f}' for m in MODELS])
    rows.append(['average longest carry chain (places)'] + [f'{MEAN_CHAIN[m]:.1f}' for m in MODELS])
    for t in ('test_indist', 'test_gap', 'test_hint'):
        rows.append([f'first round at 99%, {TLAB[t]}'] + [str(settle(m, t)[0]) for m in MODELS])
    return table('When the answer appears', ['', 'v22', 'v23', 'v33'], rows)


def t_chain():
    rows = []
    for m in MODELS:
        c = CS[m]
        rows.append([m, f'{c["org"]:.1f}', f'{c["inter"]:.1f}', f'{c["term"]:.1f}', f'{A2[m]:.2f}'])
    return table(f'First round the right digit appears and stays, along {CS["v22"]["n"]} crafted carry chains',
                 ['', 'carry starts', 'inside', 'carry ends', 'fit, rounds/place'], rows)


def t_drift():
    rows = []
    for m in MODELS:
        for fam, nm in (('nocarry_10', 'no carries'), ('nines_24', '24 nines + 1')):
            e = E(m, fam)
            rows.append([f'{m}, {nm}', f'{e["r12"]:.3f}', f'{e["r24"]:.3f}', f'{e["r48"]:.3f}', f'{e["last"]:.3f} (r{e["nlast"]})', f'{e["cos"]:.2f}'])
    return table('Size of the change in the state per round, relative to the state (place tokens)',
                 ['', 'round 12', 'round 24', 'round 48', 'last round', 'cosine'], rows, cls='compact')


def t_extra():
    rows = []
    for t in TIERS:
        cells = [TLAB[t]]
        for m in MODELS:
            c = curve(m, t)
            cells += [cell(c[BUDGET[m]], 0.99, 0.5, pct), cell(c[BUDGET[m] + EXTRA], 0.99, 0.5, pct)]
        rows.append(cells)
    rows.append({'cls': 'split', 'cells': ['problems per length'] + [str(NS[m]) for m in MODELS for _ in (0, 1)]})
    return table('Accuracy at the trained round budget, then 40 rounds later',
                 ['', 'v22 @24', '@64', 'v23 @24', '@64', 'v33 @80', '@120'], rows, cls='compact')


def t_ablate():
    cols = [('test_indist', '1–10'), ('test_hint', '14–18'), ('test_ood', '19–21'), ('test_far_ood', '24–26'),
            ('nines_8_16_22', 'runs of nines')]
    rows = [['nothing removed'] + [cell(B1['baseline'][c], 0.99, 0.5, pct) for c, _ in cols]]
    for kind, nm in (('zero', 'zeros'), ('mean', 'others&rsquo; mean')):
        for j in range(4):
            rows.append({'cls': 'split' if j == 0 else '', 'cells': [f'token {j} &rarr; {nm}'] + [cell(B1[f"{kind}_{j}"][c], 0.99, 0.5, pct) for c, _ in cols]})
    return table('Accuracy when one scratch token&rsquo;s starting value is replaced (200 problems per column)',
                 [''] + [n.replace('runs of nines', 'nines') for _, n in cols], rows)


def t_limits():
    rows = []
    for d in LD:
        rows.append([f'{d}', cell(LN[str(d)], 0.99, 0.5), cell(LC[str(d)], 0.99, 0.5), cell(LN2[str(d)], 0.99, 0.5), cell(LC2[str(d)], 0.99, 0.5)])
    return table('v33, 200 problems per row (the longer operand has this many digits)',
                 ['digits', 'random, 80 rounds', 'carry-heavy, 80', 'random, 140', 'carry-heavy, 140'], rows)


# ---------------------------------------------------------------------------------------------------------
# page

n_ood = NS['v22']
body = f'''
<div class="page">

  <div class="kicker">research notebook &middot; mechanistic interpretability</div>
  <h1 class="title">What the Adder Learned</h1>
  <p class="dek">A small transformer, built for the job, adds numbers a few digits longer than any it was trained on. <a href="https://ljtyk.github.io/carry-the-one/">Carry the One</a> asked how to build it. This post opens it up. Recording its state after every round of computation, probing what each digit position knows, following attention, and swapping states between runs gives a consistent picture: every digit does its own arithmetic at once, a carry then moves along the number one place at a time, and the answer slowly settles. The same measurements explain two of its failures and pointed to a training change that removes one of them. A final test shows that change moved the limit; it did not remove it.</p>
  <div class="byline">
    <span>research log</span>
    <span>three checkpoints, 1.8M parameters each</span>
    <span>written with Claude (Anthropic) as research assistant</span>
  </div>

  <div class="stat-duel">
    <div class="stat-cell fail">
      <div class="label">v22 &middot; run 40 rounds past the 24 it was trained for</div>
      <div class="num">{pct(DUEL_L)}</div>
      <div class="sub">exact match on 19&ndash;21 digit problems after 64 rounds ({NS['v22']} random problems). At round 24, on the same problems: {pct(DUEL_L0)} (100% on the full test set). The answer is there, then it drifts away.</div>
    </div>
    <div class="stat-cell ok">
      <div class="label">v33 &middot; run 40 rounds past the 80 it was trained for</div>
      <div class="num">{pct(DUEL_R)}</div>
      <div class="sub">same problems after 120 rounds ({NS['v33']} random problems), and every other length tested. Trained with a random number of rounds per batch and, in this model, longer problems.</div>
    </div>
  </div>

  <p class="lede">The model from Carry the One reads two numbers, writes each into a fixed set of slots, repeats one transformer block over those slots, and reads the sum back out. It was trained on operands of up to 20 digits and is exact at 21; what it was not asked to do is explain itself. Its computation is small enough to take apart: one block, repeated, with a handful of numbers on every digit position at every round. I wanted to know what the block is doing in those rounds, and whether knowing that would say anything useful about where the model fails.</p>

  <h2>Findings at a glance</h2>
  <ol class="steps">
    <li><strong>Time to a correct answer tracks the number of digits, not the longest carry chain</strong> (correlation {A4['v22']['corr_convround_vs_digits']:.2f} against {A4['v22']['corr_convround_vs_chainlen']:.2f} in v22).</li>
    <li><strong>Each place computes in a fixed order</strong>: its own digit sum and its own carry-out within the first two rounds, then whether a carry arrives, then the final digit. The place that is right last is usually where a carry chain ends ({CS['v22']['last_is_term']} of {CS['v22']['n']} crafted chains; in the rest, where it starts).</li>
    <li><strong>The carry travels one place at a time.</strong> Attention concentrates on the next place down (about {pct(SIDE['offset_profile_round6'][BINS.index(-1)])} of a place&rsquo;s attention in a 16-digit problem). Along a run of nines a carry advances about {F4['slope_rounds_per_place']:.2f} rounds per place. That is linear in the length of the chain; I found no sign of carry-lookahead.</li>
    <li><strong>The carry is a real, readable quantity.</strong> Swapping one place&rsquo;s state from a run where a carry arrives into a run where it does not changes that place&rsquo;s answer digit by exactly one (swapped in at round 14, {pct(F1['14']['flip_rate'])} of 240 problems), and changes nothing else.</li>
    <li><strong>The four scratch tokens are not interchangeable.</strong> Replacing token 0 breaks every length; token 2 can be zeroed with no effect; token 3 matters more the longer the problem. Together they carry the problem&rsquo;s length ({pct(B3['digit_count']['concat']['8'])} decodable) but not its carry statistics.</li>
    <li><strong>The model never reaches a fixed point.</strong> v22&rsquo;s state keeps moving in one direction after the answer is right, and, depending on the length of the problem, somewhere between round 25 and round 50 it leaves the answer behind. Training with a random number of rounds removes that within the lengths it was trained on.</li>
    <li><strong>The fix has a boundary.</strong> v33 is at {p1(min(v for d, v in list(LN.items()) + list(LC.items()) if int(d) <= 64))} or better through 64 digits (its training limit), {p1(LN['72'])} at 72 and {p1(LN['80'])} at 80. Running it 60 rounds longer changes nothing, so the limit is how long a problem the model was trained on, not how long it computes.</li>
  </ol>

  <div class="ledger"></div>

  <h2>The model</h2>
  <p>Numbers go in as digits, one per place, units first. Each digit becomes three numbers (one of ten learned &ldquo;codewords&rdquo;) written into that place&rsquo;s own slot, so place 7 always means the same three of 384 dimensions no matter how long the number is, and short numbers are padded with zeros out to 30 places. The <strong>combiner</strong> then turns each place&rsquo;s two codewords into a vector (a &ldquo;place token&rdquo;), adds four learned <strong>scratch tokens</strong> (called thinking tokens in the code), and applies one transformer block &mdash; six-head attention that every token can use on every other, then an MLP &mdash; again and again with the same weights. After the last round each place&rsquo;s vector is cut back to three numbers and read as the nearest codeword. The last step has no parameters, and the first has about forty; the arithmetic is all in the repeated block.</p>
  <div class="diagram">{DIAGRAM}</div>
  <p>The three checkpoints used here, all from the same family, all trained on freshly generated problems (30% built to contain long carry chains):</p>
  <ul class="points">
    <li><strong>v22</strong>, the &ldquo;closed answer&rdquo; of Carry the One: 24 rounds, operands up to {TRAIN_MAX['v22']} digits, 30 place slots. Checkpoint from iteration {dyn['v22']['iter']:,}.</li>
    <li><strong>v23</strong>: v22 continued with a random number of rounds on every batch (anywhere from as many rounds as the problem has digits up to 24) instead of always 24. Iteration {dyn['v23']['iter']:,}.</li>
    <li><strong>v33</strong>: the same idea with up to {BUDGET['v33']} rounds, 128 place slots, and operands up to {TRAIN_MAX['v33']} digits, continued from v22. Iteration {dyn['v33']['iter']:,}.</li>
  </ul>
  {t_tiers()}
  <p class="fig-cap">Full test sets. v23 and v33 were evaluated on a CPU in 32-bit floats; v22&rsquo;s numbers are from the original GPU run, and re-running it on the CPU gives identical numbers (both logs are in data/logs). v22 and v23 were trained on operands up to 20 digits, so the 24&ndash;26 digit set is four to six digits past their limit; v33 was trained on up to 64, so every length in this table is inside its range.</p>

  <div class="ledger"></div>

  <h2>How the model was opened</h2>
  <p>Everything below rests on a few small instruments. I added three inert hooks to the model: one records the full state after every round, one lets a function rewrite the state between rounds, and one records the attention weights. Every measurement below was run on a CPU.</p>
  <ul class="points">
    <li><strong>Reading the answer early.</strong> The model&rsquo;s own readout is applied to the state after round <em>k</em> for every <em>k</em>: &ldquo;what would the answer be if we stopped here?&rdquo; Round 0 is the state before the first round. v22 was never trained to be read before its last round, so early readouts are rough; they are measurements of the state, not of intent.</li>
    <li><strong>Linear probes.</strong> A least-squares classifier (ridge, fixed penalty) on a place token&rsquo;s 384 numbers, trained on 70% of 500 problems and scored on the other 30%, split by problem. The 500 problems have 2&ndash;18 digits: a quarter with two operands of the same length, a quarter of different lengths, half with a run of nines. Each result is compared against always guessing the most common answer.</li>
    <li><strong>Attention maps</strong> from the block, averaged over its six heads and over 24 problems of each kind.</li>
    <li><strong>State swaps</strong> (activation patching, as in causal tracing): replace one place&rsquo;s state at one round with the state from a second run on slightly different operands, and see what the answer does.</li>
    <li><strong>Ablations</strong> of the scratch tokens: set a token&rsquo;s starting value to zeros, or to the average of the other three.</li>
  </ul>
  <p>Rounds are counted from 1 for the first application of the block. Where a chart says &ldquo;round 0&rdquo; it means the state before any round. The fixed test sets and the length test (last section) use one operand of the stated length and the other of any length up to it. In the attention maps and scratch-token probes, &ldquo;random&rdquo; means two operands of the stated length and &ldquo;nines&rdquo; means a number ending in a run of nines plus a shorter random number. &ldquo;Carry-heavy&rdquo; problems are built the way the training generator builds them, with a long run of nines in one operand and a nonzero units digit in the other.</p>

  <div class="ledger"></div>

  <h2>The time it takes tracks length, not the carry chain</h2>
  <p>For each problem, find the first round after which the model&rsquo;s answer stops changing. Across random problems that round correlates almost perfectly with the number of digits ({A4['v22']['corr_convround_vs_digits']:.2f} in v22) and weakly with the longest carry chain in the problem ({A4['v22']['corr_convround_vs_chainlen']:.2f}). The chains are short on average ({MEAN_CHAIN['v22']:.1f} places in this sample), so the model is mostly not waiting for a long carry to arrive. One reading is that it is waiting to be sure there is none; I did not test that.</p>
  {t_timing()}
  <p class="fig-cap">Round-by-round probe, {NS['v22']} random problems per length set for v22 and v23, {NS['v33']} for v33. &ldquo;Stops changing&rdquo; is measured up to the trained round budget.</p>
  <p>Crafted carry chains show where the time goes. Take a number with a run of nines and add one at the bottom, so a carry must pass through every nine. The places inside the chain are right almost at once ({CS['v22']['inter']:.1f} rounds on average in v22). The place where the carry starts takes {CS['v22']['org']:.1f}, and the place where it ends takes the longest, {CS['v22']['term']:.1f}. In all {CS['v22']['last_is_end']} chains one of those two ends was the last place to settle: the end of the chain in {CS['v22']['last_is_term']}, the start in the others.</p>
  {t_chain()}
  <p class="fig-cap">Chains of 1, 3, 6 and 10 nines starting at place 0, 4 or 9. The last column is a straight-line fit of round against place over all places the carry touches.</p>
  <div class="note">
    <div class="note-label">A trap in the obvious test</div>
    <p>The simplest carry test, 99&hellip;9 + 1, is answered correctly by round {min(v['first_all_correct_round'] for v in dyn['v22']['A1_nines_ripple'].values())}&ndash;{max(v['first_all_correct_round'] for v in dyn['v22']['A1_nines_ripple'].values())} in v22 for every length from 4 to 24 digits, which would look like carries that cost nothing. It is not. The training generator produces exactly that pattern (a full run of nines is one of its cases), and a run of nines in the <em>middle</em> of a number is a much slower case, below. I read the 99&hellip;9 + 1 result as a learned fast path for one input, not as evidence about carries in general.</p>
  </div>

  <div class="ledger"></div>

  <h2>What each place computes, and in what order</h2>
  <p>A linear probe on a place token at each round shows an order of operations. Before any round the token holds just its two digits; (a + b) mod 10 is not yet readable ({pct(P_R('sum_nc', 0))}, against {pct(base['sum_nc'])} for always guessing the commonest digit). One round later it is ({pct(P_R('sum_nc', 1))}). The carry-out is mostly decided by the place&rsquo;s own two digits and is readable almost immediately. Whether a carry <em>arrives</em> takes several rounds, and the final digit, which needs all of it, comes last.</p>
  {chart_probe}
  <p class="fig-cap">v22. Held-out accuracy of a linear probe on a place token, from {carry['n_examples']} problems (the mix described above). Always guessing the commonest answer scores {pct(base['sum_nc'])} for the digit sum, {pct(base['carry_out'])} for carry-out and carry-in, and {pct(base['final'])} for the answer digit. By round 8 the digit sum is read correctly for {p1(P_R('sum_nc', 8))} of places and carry-in for {p1(P_R('carry_in', 8))}; the answer digit reaches {pct(P_R('final', 12))} by round 12 and {p1(P_R('final', 16))} by round 16.</p>
  <p>The picture sharpens when places are sorted by what the carry does to them. A place inside a chain, or with no carry, is already {pct(min(PT_R('final', 'chain_interior', 2), PT_R('final', 'no_carry', 2)))}&ndash;{pct(max(PT_R('final', 'chain_interior', 2), PT_R('final', 'no_carry', 2)))} readable at round 2. A place where a carry starts is {pct(PT_R('final', 'chain_origin', 2))} readable at round 2 and {pct(PT_R('final', 'chain_origin', 8))} at round 8. A place where a carry ends is the slowest: its answer digit is not readable at all at round 0 and only {pct(PT_R('final', 'chain_terminus', 4))} readable at round 4. All four kinds are above 99% by round 16.</p>
  <div class="fig-title">answer digit, by kind of place</div>
  {chart_type_final}
  <p>The carry-in probe is more telling. At the end of a chain the true carry-in is always 1, yet at round 1 the probe says 0 for almost every place ({pct(PT_R('carry_in', 'chain_terminus', 1))} right, against {pct(base['carry_in'])} for guessing). Those places are not uncertain; their state confidently says &ldquo;no carry&rdquo;, and it takes the carry arriving from below to correct it.</p>
  <div class="fig-title">carry-in, by kind of place</div>
  {chart_type_cin}
  <p class="fig-cap">v22. &ldquo;No carry&rdquo; and &ldquo;inside a carry&rdquo; places are right early; the two ends of a chain are not. Places beyond the length of the number (not drawn) are right from round 0.</p>

  <div class="ledger"></div>

  <h2>The carry moves one place at a time</h2>
  <p>Attention says how. At round 7 the strongest attention in the block goes to the next place <em>down</em>, in all eight kinds of problem tried (random and runs-of-nines, 6 to 20 digits). In a 16-digit random problem {pct(SIDE['offset_profile_round6'][BINS.index(-1)])} of a place&rsquo;s attention goes there, against at most {pct(max(SIDE['offset_profile_round6'][i] for i, b in enumerate(BINS) if b != -1 and abs(b) <= 4))} for any other single place within four. That is the place its carry-in comes from, and the direction is one-way: per place, {pct(SIDE['flow_down_by_round'][6])} of attention goes to lower places and {pct(SIDE['flow_up_by_round'][6])} to higher ones.</p>
  {chart_off}
  <p class="fig-cap">v22. Share of a place token&rsquo;s attention by how far, in places, the attended token is from it, averaged over six heads and 24 problems, restricted to places inside the number. On a run of nines the band widens to the next two or three places down while the carry is in transit (round 7), then narrows again (round 17).</p>
  <p>The four scratch tokens behave differently. In round 1 place tokens look mainly at themselves ({pct(SIDE['self_attn_by_round'][0])} of their attention). In rounds 2 to 4 they send {pct(th_rand16[1])}, {pct(th_rand16[2])} and {pct(th_rand16[3])} of it to the scratch tokens (16-digit problems), and the scratch tokens send {pct(SIDE['think->place_by_round'][1])} of theirs back to the place tokens in round 2. By round 11 the place-to-scratch share has fallen to {pct(th_rand16[10])}. On long inputs it rises again late: to {pct(th_nines20[20])} on 20-digit runs of nines at round 21, after dipping to {pct(min(th_nines20[10:18]))}.</p>
  {chart_scr}
  <p class="fig-cap">v22. Share of a place token&rsquo;s attention that goes to the four scratch tokens.</p>

  <div class="ledger"></div>

  <h2>Is the carry a real variable?</h2>
  <p>Probes and attention describe; swapping states tests. Take a 14-digit problem where some place <em>p</em> has no carry coming in and whose two digits sum to 2&ndash;6 (so one more cannot carry on), and make a second run in which the place below <em>p</em> is forced to produce a carry. At some round, copy the second run&rsquo;s state for place <em>p</em> into the first run and carry on. If <em>p</em>&rsquo;s state really holds the carry, its answer digit should go up by exactly one and nothing else should move.</p>
  {chart_f1}
  <p class="fig-cap">v22, 240 problems, <em>p</em> between 4 and 10. Swaps before round 6 do nothing, presumably because the second run has not yet worked out its own carry. By round 14 every swap raises the digit by one, none changes the next place up, and no other place changes. As a control, swapping the state of a place three below <em>p</em> at round 4, where the two runs agree, left every answer unchanged.</p>
  <p>So by round 10 or so a place&rsquo;s state carries the carry-in in a form the readout uses. The negative results matter as much. The same swap into the bottom of a run of nines at rounds 12, 14 and 16 does flip that place, but none of the six places above it ever followed (at rounds 8 and 10 not even that place flipped). A second attempt added a &ldquo;carry-in direction&rdquo; (the difference between the average states with and without a carry, at 0.5, 1 and 2 times a token&rsquo;s typical size) at rounds 3, 5, 8 and 12; no place in the run flipped, in any of the 160 problems, in any setting. The carry cannot be injected mid-way and have the block carry it on. <strong>It behaves like a trajectory that unfolds from the real operands, not a procedure that can be re-entered.</strong> That is a statement about these two ways of injecting a carry, not a proof that no other way exists.</p>
  <p>Without any patching, the natural speed of a carry can be read from paired runs: a number with a run of nines, with and without a 1 added at the bottom, decoded at every round.</p>
  {chart_f4}
  <p class="fig-cap">v22, {int(round(F4['carry_ok'] * 200))} of 200 problems carried all the way; runs of 7&ndash;11 nines in the middle of a 14-digit number, each place&rsquo;s digit first becoming 0 and staying there. The unpatched controls decoded as a clean run of nines in {pct(F4['base_ok'])} of cases. The first three places behave irregularly; from place 3 on the ramp is straight.</p>
  <p>The fit gives {F4['slope_rounds_per_place']:.2f} rounds per place over all places, {F4_SLOPE_HI:.1f} from place 3 on, with about 5 rounds of start-up. At that speed 24 rounds reach about the tenth place of a chain, so a long mid-number carry does not fit in v22&rsquo;s budget, and the cost is linear in the length of the chain. A carry-lookahead adder would be logarithmic. I did not find one here. The wavefront of attention at offset &minus;1 is what a ripple looks like once all the places run at once.</p>

  <div class="ledger"></div>

  <h2>The four scratch tokens</h2>
  <p>The scratch tokens are learned vectors added to the sequence with no digit attached. They are not interchangeable, and not all of them matter.</p>
  {t_ablate()}
  <p class="fig-cap">v22. One token&rsquo;s starting value replaced by zeros, or by the average of the other three (&ldquo;others&rsquo; mean&rdquo;). &ldquo;Nines&rdquo; are problems with a run of nines, of 8, 16 and 22 digits. Replacing all four tokens by either also gives 0% everywhere. Token 0 cannot be replaced by either: the model gets nothing right, at any length. Token 2 can be removed with no loss (accuracy on the 24&ndash;26 digit set was unchanged or a little higher, within the noise of 200 problems). Token 1 is hurt less by the average of the others than by zeros. Token 3 costs more the longer the problem.</p>
  <p>Attention is consistent with that. In the first ten rounds token 0 receives the most attention from place tokens (up to {pct(max(R16[0]['received_from_place']))} of a place token&rsquo;s attention on a 16-digit problem, against at most {pct(max(max(R16[j]['received_from_place'][:10]) for j in (1, 2, 3)))} for the others). From round 4 to 9 on 20-digit runs of nines, token 3 receives the most of the four (up to {pct(max(N20[3]['received_from_place']))}). Token 1 reads the place tokens as heavily as the others do, but is almost never read by them (at most {pct(max(max(R16[1]['received_from_place']), max(N20[1]['received_from_place'])))}).</p>
  <p>What do they hold? A linear probe on the four tokens together reads the <strong>number of digits</strong> in the problem almost perfectly by round 4 ({pct(B3['digit_count']['concat']['4'])}; always guessing the commonest length would score {pct(B3_BASE['digit_count'])}), and reads the total number of carries ({pct(B3['total_carry']['concat']['8'])} at round 8, against {pct(B3_BASE['total_carry'])}) and the longest chain ({pct(B3['max_chain']['concat']['8'])}, against {pct(B3_BASE['max_chain'])}) barely better than guessing. They carry the length of the problem, not its carry statistics. I read &ldquo;shared channel for the length&rdquo; into that; it is an interpretation, and I did not open the circuits.</p>

  <div class="ledger"></div>

  <h2>The answer never settles</h2>
  <p>Run v22 longer than the 24 rounds it was trained for and accuracy does not stay put:</p>
  {CHART_ACC['v22']}
  <p class="fig-cap">v22, {NS['v22']} random problems per length set; curves are means over those problems, so heights differ from the full test sets by a few points (the 19&ndash;21 digit set reads {pct(DUEL_L0)} at round 24 here and 100% on all 600 test problems). The shapes are the point.</p>
  <p>Each length set rises, plateaus, and falls, later for shorter problems. Longer problems need more rounds to reach their plateau: the 14&ndash;18 digit set first reaches 99% at round {settle('v22', 'test_hint')[0]}, against round {settle('v22', 'test_indist')[0]} for 1&ndash;10 digits. The 24&ndash;26 digit set peaks at round {curve('v22', 'test_far_ood').index(max(curve('v22', 'test_far_ood')))} with {pct(max(curve('v22', 'test_far_ood')))}, just past the trained 24, and is at {pct(curve('v22', 'test_far_ood')[BUDGET['v22'] + EXTRA])} by round 64. This is why the readout round matters, and it fits a simple reading of v22&rsquo;s weakness at 25&ndash;26 digits (80% and 52%): one or two more rounds would help, and then the drift would take it back. That reading is consistent with the curve; I did not run an intervention that isolates it.</p>
  <p>The state itself says why. Track how much it changes per round, relative to its size, and compare consecutive changes by direction:</p>
  {t_drift()}
  <p class="fig-cap">Place tokens, averaged over 24 problems of each kind (one problem for the nines). The cosine compares each round&rsquo;s change with the previous round&rsquo;s (late rounds): 1.00 means the state keeps being pushed the same way. The last round the probe ran is 64 for v22 and v23, and 120 for v33.</p>
  <p>The state does not settle to a point. After the answer is already right it keeps travelling in one direction at a small but non-zero rate: about {pct(E('v22', 'nocarry_10')['last'])}&ndash;{pct(E('v22', 'nocarry_10')['r24'])} of its own size per round between rounds 24 and 64 (more, {pct(E('v22', 'nines_24')['r24'])}, for the 24-nines problem at round 24, where the carry is still in transit). For a while that is invisible because the decision boundaries are far away; in v22 it eventually crosses them. Block weights shared across rounds do not on their own give you an equilibrium (compare deep-equilibrium models, which solve for one); they give you a trajectory, and a trajectory is read at a time.</p>
  <p>v23 changes only how the block is trained: each batch uses a random number of rounds, from as many as the problem has digits up to the full budget, so the block is regularly asked to keep a right answer right.</p>
  {CHART_ACC['v23']}
  <p class="fig-cap">v23, {NS['v23']} random problems per length set.</p>
  <p>The shorter lengths now hold flat to round 64, and the answer arrives earlier (mean round {A4['v23']['mean_conv_round']:.1f}, against {A4['v22']['mean_conv_round']:.1f}; on 1&ndash;10 digit problems round {settle('v23', 'test_indist')[0]} against {settle('v22', 'test_indist')[0]}). The state still drifts at about the same rate; it just does not reach a decision boundary. The cost is real. On the 24&ndash;26 digit set v23 is <strong>worse</strong> than v22 ({p1(acc('v23', 'test_far_ood'))} against {p1(acc('v22', 'test_far_ood'))}), and still falls with extra rounds. A model that converges fast and holds is a model that holds a wrong answer if a long carry has not finished.</p>
  <p>v33 combines that training with an 80-round budget and operands up to 64 digits:</p>
  {CHART_ACC['v33']}
  <p class="fig-cap">v33, {NS['v33']} random problems per length set, run to 120 rounds. All five length sets are at 100% on this sample from the round they first reach 99% (round {min(settle('v33', t)[0] for t in TIERS)}&ndash;{max(settle('v33', t)[0] for t in TIERS)}) to round 120.</p>
  {t_extra()}
  <p class="fig-cap">Every model run 40 rounds past its budget. v33 is the only one that is flat on all five.</p>
  <p>It is exact on the 24&ndash;26 digit set ({p1(acc('v33', 'test_far_ood'))} of 600, against {p1(acc('v22', 'test_far_ood'))} for v22) and trades a sliver of the shorter sets for it ({p1(acc('v33', 'test_indist'))} on 1&ndash;10 digits). Two cautions belong here. First, v33 differs from v23 in three ways at once: more rounds, longer training problems, and 128 instead of 30 place slots. v23 alone shows that variable round counts do not by themselves fix the 24&ndash;26 digit set; I did not run the other additions on their own, so I cannot say which of them is needed. Second, those 24&ndash;26 digit problems are inside what v33 was trained on. Whether they are &ldquo;extrapolation&rdquo; is the next section.</p>
  {t_far()}

  <div class="ledger"></div>

  <h2>Where the fix runs out</h2>
  <p>Carry the One says that raising the training range moves the wall rather than removing it: v33, trained to 64 digits, is exact through 26 only because 26 is inside its range. Here is the direct test. I generated fresh problems at lengths from 24 to 124 digits (the model has 128 place slots), 200 per length, random and carry-heavy.</p>
  {chart_lim}
  <p class="fig-cap">v33. Dotted lines are the same model run for 140 rounds instead of 80; they sit on top of the solid ones.</p>
  {t_limits()}
  <p>v33 is at {p1(min(LN['64'], LC['64']))} or better at its training limit of 64 digits, still at {p1(LN['72'])} (random) and {p1(LC['72'])} (carry-heavy) at 72, eight digits (12.5%) past it, and falls off after that: {p1(LN['80'])} at 80, {p1(LN['96'])} at 96. Random and carry-heavy problems fall together up to about 96 digits, so up to there this is length, not carries. Past it carry-heavy problems collapse much faster ({p1(LC['112'])} against {p1(LN['112'])} at 112 digits).</p>
  <p><strong>Giving the model 60 more rounds changes nothing</strong>: no length moves by more than {max(max_diff_140) * 100:.1f} points. By round 80 the model has finished and then holds its answer, which for these lengths is wrong. The limit is not how long the block runs. It is how long a problem the model was trained on, which is the claim in Carry the One, now measured.</p>
  <p>The same shape shows up in both models. v22 was trained to 20 digits and v33 to 64. In each, accuracy starts to break about a quarter past the longest training problem: v22 is {p1(far_by_digit['v22']['24'])} at 24 digits (+20%), {p1(far_by_digit['v22']['25'])} at 25 (+25%) and {p1(far_by_digit['v22']['26'])} at 26 (+30%); v33 is {p1(LN['72'])} at 72 (+12.5%) and {p1(LN['80'])} at 80 (+25%). Two models are a coincidence, not a law, but it is consistent with a model whose computation is sound and whose sense of &ldquo;how long a number can be&rdquo; ends where its training did. As a back-of-envelope, {p1(LN['80'])} exact on an 80-digit answer is about 99.6% per digit if errors were independent; the model is nearly right everywhere, and exact match is unforgiving at length.</p>

  <h2>Limits</h2>
  <ul>
    <li>One task, one model family, <strong>one training run per checkpoint</strong>. No seed variation; the differences between v22, v23 and v33 are single runs.</li>
    <li>The probes are small. Linear probes use 500 problems (150 held out) and a fixed ridge penalty, so their accuracies carry a few points of noise. The round-by-round accuracy curves use {NS['v22']} problems per length set ({NS['v33']} for v33), enough for shapes, not for the third digit of any percentage. Early-round readouts apply a readout the model was not trained to be read at.</li>
    <li>Attention is averaged over six heads, and I did not open the MLP. The &ldquo;scratch-token roles&rdquo; are read from ablations, attention and probes, not from a traced circuit.</li>
    <li>The state-swap result is positive for one kind of swap (a whole place state, after round 10) and negative for two others. The negative results do not show the carry cannot be injected, only that these two ways failed.</li>
    <li>The wavefront speed comes from one family of inputs (runs of 7&ndash;11 nines in a 14-digit number), and {round(100 * (1 - F4['base_ok']))}% of its unpatched controls were not decoded as a clean run of nines.</li>
    <li>v33 changed three things at once from v22/v23, and the variable-round change was the only one run alone. I make no claim about which of the other two it needs.</li>
    <li>For v33 the training script&rsquo;s own &ldquo;best checkpoint&rdquo; (by a clipped validation loss) stopped at iteration 3000 and did not track accuracy. Every v33 number here is from the final iteration-15,000 snapshot.</li>
    <li>In <code>v33_dynamics.json</code> the keys <code>acc_at_24</code> and <code>acc_at_64</code> mean &ldquo;at the model&rsquo;s own round budget&rdquo; and &ldquo;at budget + 40&rdquo; (80 and 120 for v33). The charts and tables here are computed from the full per-round curves, not from those keys.</li>
  </ul>

  <h2>What I would try next</h2>
  <p>All of this is untested. The training-length result suggests the model needs a reason not to treat the longest length it saw as the longest possible. Two ideas follow from the measurements. One is to remove the fixed ceiling: draw the maximum length per batch from a wide range, so no consistent &ldquo;answers stop here&rdquo; is ever learned. The other comes from the attention maps: only the next place down and the scratch tokens carry much attention, yet every place attends over every other place, so a place&rsquo;s share of attention to its neighbour could fall as numbers get longer. Within the lengths measured here (6 to 20 digits, v22) it does not obviously do so, but I have not measured it past the training length, which is the first thing to check. If it does fall, restricting the block to a local window plus the scratch tokens would make that share independent of length. A caution: in the single-token experiments of the companion post, restricting attention to a local band did not sharpen it, so I would not assume it carries over. A cleaner test of the training-length result is to vary the training maximum alone, holding everything else at v33.</p>

  <div class="verdict">
    <div class="note-label">Where this leaves things</div>
    <h3>The block is doing sound arithmetic. It does not know when it should stop being right.</h3>
    <p>Taking the model apart shows an algorithm anyone would recognise: each digit does its own sum, a carry moves up the number a place at a time, and the answer settles. It also shows why the model fails: a state that drifts if read too late, a round budget that is tight for the longest problems, and a training length that sets the edge of the numbers it trusts. One of those is fixed by a change in how it is trained. The edge moved with the training range, in both models; it did not go away.</p>
  </div>

  <h2>References</h2>
  <ol class="refs">
    <li>M. Dehghani et al., <em>Universal Transformers</em>, ICLR 2019 (arXiv 1807.03819). Repeating one block, which is what the combiner does.</li>
    <li>S. Bai, J. Z. Kolter, V. Koltun, <em>Deep Equilibrium Models</em>, NeurIPS 2019 (arXiv 1909.01377). Solving for a fixed point instead of unrolling a fixed number of rounds.</li>
    <li>N. Nanda et al., <em>Progress measures for grokking via mechanistic interpretability</em>, ICLR 2023 (arXiv 2301.05217). Taking apart a small transformer that adds.</li>
    <li>K. Meng et al., <em>Locating and Editing Factual Associations in GPT</em>, NeurIPS 2022 (arXiv 2202.05262). Causal tracing, the state-swap idea.</li>
    <li>F. Zhang, N. Nanda, <em>Towards Best Practices of Activation Patching in Language Models</em>, ICLR 2024 (arXiv 2309.16042).</li>
    <li>H. Cho et al., <em>Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure</em>, NeurIPS 2024 (arXiv 2405.20671). The other route to length generalization in addition.</li>
  </ol>

  <footer>
    every number in this post comes from the result files in <a href="https://github.com/ljtyk/adder-mech-interp/tree/main/data">data/</a>; the code behind them is in <a href="https://github.com/ljtyk/adder-mech-interp/tree/main/code">code/</a>. companion posts: <a href="https://ljtyk.github.io/carry-the-one/">Carry the One</a> &middot; <a href="https://ljtyk.github.io/rope-aliasing/">Why Rotary Embeddings Stop Extrapolating</a>
  </footer>

</div>
'''

page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>What the Adder Learned</title>
<meta name="description" content="Mechanistic interpretability of a small transformer that adds numbers at lengths it was never trained on: what each digit position computes, how a carry travels, why the answer drifts, and where the fix runs out.">
<style>
{base_css}
{EXTRA_CSS}
</style>
</head>
<body>
{body}
</body>
</html>
'''
open(os.path.join(HERE, 'index.html'), 'w', encoding='utf-8').write(page)
print('wrote index.html', len(page), 'bytes')
print('chain stats v22:', CS['v22'])
print('F4 slope all / k>=3:', F4['slope_rounds_per_place'], round(F4_SLOPE_HI, 3))
print('B3 baselines:', B3_BASE)
