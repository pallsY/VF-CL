"""3-seed mean+/-std summary for c10 cells, sorted by a forget+retain score."""
import json, glob, os

def load(cell):
    rows = []
    for f in glob.glob(f'results/frag/{cell}/*/**/aggregated.json', recursive=True):
        d = json.load(open(f))
        cl, ul = d.get('cl_method', '?'), d.get('ul_method', '?')
        clm, ulm = d.get('cl_metrics', {}), d.get('ul_metrics', {})
        def ms(dct, k):
            v = dct.get(k)
            if isinstance(v, dict):
                return v.get('mean'), v.get('std')
            return None, None
        AA, AAs = ms(clm, 'AA'); AAf, _ = ms(clm, 'AA_final')
        fa, fas = ms(ulm, 'forget_acc'); ra, ras = ms(ulm, 'retain_acc'); mi, mis = ms(ulm, 'mia_score')
        rows.append(dict(name=f'{cl}_x_{ul}', AA=AA, AAf=AAf, fa=fa, fas=fas,
                         ra=ra, ras=ras, mi=mi, n=d.get('n_seeds')))
    return rows

def f(v, s=None, p=3):
    if v is None: return '  -  '
    return f'{v:.{p}f}' + (f'±{s:.2f}' if s is not None else '')

for cell in ['c10_1p', 'c10_4p']:
    rows = load(cell)
    # sort: ROAR first, then by retain among low-forget
    rows.sort(key=lambda r: ('roar' not in r['name'], (r['fa'] or 1) > 0.1, -(r['ra'] or 0)))
    print(f'\n#### {cell}   (3 seeds; forget<=0.1 = "really forgets")')
    print(f'{"method":30} {"AA":>6} {"AAfin":>6} {"forget↓":>11} {"retain↑":>11} {"MIA":>6}')
    for r in rows:
        star = ' *' if 'roar' in r['name'] else '  '
        print(f'{r["name"][:30]:30} {f(r["AA"]):>6} {f(r["AAf"]):>6} '
              f'{f(r["fa"],r["fas"]):>11} {f(r["ra"],r["ras"]):>11} {f(r["mi"]):>6}{star}')
