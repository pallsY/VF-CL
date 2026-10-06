"""Localize LwF retention collapse: class-IL vs task-IL vs seen-IL from final_probs.npz."""
import numpy as np, glob, sys

run_glob = sys.argv[1] if len(sys.argv) > 1 else "results/after/*/final_probs.npz"
d = sorted(glob.glob(run_glob))[-1]
z = np.load(d)
probs, labels, retained = z["probs"], z["labels"], z["retained"]
print("file:", d)
print("probs shape:", probs.shape, "retained:", retained.tolist())

tasks = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]}
full_pred = probs.argmax(1)

print("\n task |   N  | classIL(full 10way) | taskIL(within-2) | seenIL(within-seen)")
seen = []
for t, cs in tasks.items():
    seen += cs
    m = np.isin(labels, cs)
    n = int(m.sum())
    cil = float((full_pred[m] == labels[m]).mean())
    sub = probs[m][:, cs]
    til = float((np.array(cs)[sub.argmax(1)] == labels[m]).mean())
    subs = probs[m][:, seen]
    sil = float((np.array(seen)[subs.argmax(1)] == labels[m]).mean())
    print(f"  t{t}  | {n:4d} |        {cil:.3f}        |      {til:.3f}     |      {sil:.3f}")

print("\n full-10-way prediction histogram (rows=true task, cols=predicted class 0..9):")
for t, cs in tasks.items():
    m = np.isin(labels, cs)
    h = np.bincount(full_pred[m], minlength=10)
    print(f"  t{t} {cs} ->", h.tolist())
