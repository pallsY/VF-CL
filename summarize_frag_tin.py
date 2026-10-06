import json, glob

cells = ["c10_1p", "c10_4p", "c100_1p", "c100_4p", "tin_1p", "tin_2p", "tin_4p"]
methods = ["oracle", "finetune_x_retrain", "der_pp_x_luv",
           "proto_evolve_x_luv", "proto_evolve_x_fedosd"]

cols = ["AA", "AA_fin", "BWT", "forget", "retain", "MIA", "time_s"]
hdr = "{:<26}".format("cell / method") + "".join("{:>8}".format(c) for c in cols)


def num(v, p=4):
    return ("{:.%df}" % p).format(v) if isinstance(v, (int, float)) else "-"


for c in cells:
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        fs = sorted(glob.glob("results/frag/%s/%s/**/aggregated.json" % (c, m),
                              recursive=True))
        label = "%s/%s" % (c, m)
        if not fs:
            print("{:<26}{:>8}".format(label, "pending"))
            continue
        d = json.load(open(fs[-1]))
        cl = d.get("cl_metrics", {})
        ul = d.get("ul_metrics", {})

        def g(dd, k):
            x = dd.get(k)
            return x.get("mean") if isinstance(x, dict) else None

        row = [num(g(cl, "AA")), num(g(cl, "AA_final")), num(g(cl, "BWT")),
               num(g(ul, "forget_acc")), num(g(ul, "retain_acc")),
               num(g(ul, "mia_score"), 3),
               num(d.get("total_time_seconds", {}).get("mean"), 0)]
        print("{:<26}".format(label) + "".join("{:>8}".format(x) for x in row))
print("=" * len(hdr))
