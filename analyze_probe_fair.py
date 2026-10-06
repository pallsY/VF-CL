"""Parse probe_report.json: fresh-class Q1 ownership + Q2 single-party probe."""
import json, sys
import numpy as np

for path in sys.argv[1:]:
    try:
        r = json.load(open(path))
    except Exception as e:
        print(f"[skip] {path}: {e}"); continue
    c = r["config"]; ds = r["decision_summary"]; q2 = r["q2"]
    print("=" * 64)
    print("CONFIG  P=%s agg=%s epochs=%s  (%s)" % (
        c["num_parties"], c["aggregation"], c.get("epochs_per_task"), path.split("/")[-2]))
    print("sanity party-emb cosine =", r["sanity"]["mean_pairwise_party_embedding_cosine"])
    prev = set(); fresh = []
    print("-- Q1 per-task drift --")
    for s in r["per_task_ownership"]:
        eff = set(s["eff_classes"]); new = sorted(eff - prev); prev = eff
        pc = s["per_class"]
        def ms(cl):
            k = str(cl)
            return pc[k]["max_share"] if k in pc else (pc[cl]["max_share"] if cl in pc else None)
        eff_ms = [ms(cl) for cl in eff if ms(cl) is not None]
        new_ms = [ms(cl) for cl in new if ms(cl) is not None]
        fresh += new_ms
        print("  task %s eff=%s eff_avg_maxshare=%.3f | fresh=%s fresh_maxshare=%s" % (
            s["task"], sorted(eff), np.mean(eff_ms), new, [round(x, 3) for x in new_ms]))
    print(">>> FRESH-class avg max-share = %.3f  (uniform=%.3f)  [FAIR Q1]" % (
        np.mean(fresh), ds["uniform_share_baseline"]))
    print(">>> final-model avg max-share = %.3f  Q1_verdict=%s" % (
        ds["final_avg_max_share"], ds["q1_verdict"]))
    print("-- Q2 single-party linear probe (final eff) --")
    print("  per-party overall acc:", [round(x, 3) for x in q2["per_party_overall_acc"]])
    mp = {int(k): v for k, v in q2["per_class_max_party_acc"].items()}
    print("  per-class BEST single-party acc:",
          {k: round(v, 3) for k, v in sorted(mp.items(), key=lambda kv: -kv[1])})
    print("  classes >0.70: %s/%s  >0.50: %s/%s  Q2_verdict=%s" % (
        q2["n_classes_max_party_above_0.70"], q2["n_classes_total"],
        q2["n_classes_max_party_above_0.50"], q2["n_classes_total"], ds["q2_verdict"]))
    het = q2.get("heterogeneity", {})
    if het:
        print("  Q2 heterogeneity: mean=%.3f std=%.3f range=[%.3f,%.3f]" % (
            het.get("max_party_acc_mean", 0), het.get("max_party_acc_std", 0),
            het.get("max_party_acc_min", 0), het.get("max_party_acc_max", 0)))
