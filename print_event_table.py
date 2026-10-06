"""Print a per-event metric table from a run's results.json (or a dir of them).
Each event (CIL or UL) is one row: overall acc + per-task acc; UL events also
show forget_acc / retain_acc / mia. Lets you eyeball how each UL perturbs the
model task-by-task.

Usage: python print_event_table.py <results.json | dir>
"""
import json, sys, glob, os


def tnum(k):
    return k.split('_')[-1]


def show(f):
    d = json.load(open(f))
    c = d.get('config', {})
    print("\n### %s x %s   data=%s seed=%s   (AA_final=%.4f)" % (
        c.get('cl_method'), c.get('ul_method'), c.get('data'), c.get('seed'),
        d.get('cl_metrics', {}).get('AA_final', float('nan'))))
    print("%-12s %7s | %6s %6s %5s | per-task acc" %
          ("event", "overall", "forget", "retain", "mia"))
    print("-" * 78)
    for s in d.get('step_results', []):
        ev = "%d_%s" % (s.get('event_idx', -1), s.get('type'))
        oa = s.get('overall_acc', 0.0)
        pt = s.get('per_task_acc', {}) or {}
        pts = "  ".join("t%s:%.2f" % (tnum(k), v) for k, v in pt.items())
        if s.get('type') == 'UL':
            ul = s.get('ul_eval', {}) or {}
            fc = s.get('forget_classes', [])
            print("%-12s %7.3f | %6.2f %6.2f %5.2f | %s   (forget %s)" % (
                ev, oa, ul.get('forget_acc', 0), ul.get('retain_acc', 0),
                ul.get('mia_score', 0), pts, fc))
        else:
            print("%-12s %7.3f | %6s %6s %5s | %s" % (ev, oa, "", "", "", pts))


if __name__ == '__main__':
    p = sys.argv[1]
    files = ([p] if p.endswith('.json')
             else sorted(glob.glob(os.path.join(p, '**', 'results.json'), recursive=True)))
    for f in files:
        show(f)
