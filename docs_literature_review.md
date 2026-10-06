# Literature review — VFL/federated class-unlearning (for ROAR-v2 redesign)

## Mechanism taxonomy (removal quality)
- **Gradient ascent (old ROAR)**: removes LEAST, damages retain. (confirmed by our data)
- **Orthogonal/projection (FedOSD, REMISVFU)**: STRONGEST removal sparing retain — erase
  direction orthogonal to retain gradient. ROAR-v2 adopts this.
- **Null-space (GPM, ICLR'21) / PCGrad surgery (NeurIPS'20)**: the math template; GPM
  null-space (SVD of retain activations) is stronger than single-vector PCGrad.
- **Influence/Fisher (Certified Removal, ICML'20)**: convex-tight, degrades on deep nets.
- **SISA / exact retrain (S&P'21)**: only provably-complete removal; cost ∝ shards touched.

## Direct competitors to position against (cite!)
- **FedOSD** (AAAI'25, arxiv 2412.20200) — our strongest baseline + ROAR-v2's template.
- **REMISVFU** (arxiv 2512.10348) — split-VFL, encoder-output→random-anchor erasure +
  orthogonal projection. CLOSEST VFL competitor (already does "VFL orthogonal erasure").
- **Forgetting Any Data at Any Time** (arxiv 2502.17081) — "first certified VFL unlearning",
  ASYNCHRONOUS comm-efficiency (only requester+active party). The comm story to beat.
- **Privacy-Guaranteed Label Unlearning in VFL** (arxiv 2410.10922) — CLOSEST to our S*
  claim: "unlearning occurs only at relevant parties holding affected data."
- **Unified CL+Unlearning** (arxiv 2408.11374, 2505.15178) — centralized "learn so future
  unlearning is cheaper" via gradient/weight-saliency decomposition. CLOSEST to our CL leg.
- **Verifying Robust Unlearning: Probing Residual Knowledge** (arxiv 2504.14798) — the
  probing/relearning-attack eval our certificate lives in.

## Novelty verdict (HONEST — narrows our claims)
- **"Localized unlearning on a party subset"** = NOT a clean first (SISA, async-VFL, and
  esp. 2410.10922 already localize to relevant parties). Our genuine novelty = **CERTIFIED
  ownership-based minimal-set SELECTION** (exact per-party logit decomposition certifies
  WHICH parties own a class + a residual-leakage bound) — frame it as that, NOT "localized
  unlearning".
- **"Beat FedOSD on removal"** = we CANNOT (fundamental: touching only S* leaves evidence
  in un-touched parties; reconnection/probing attacker reads all parties). Honest frame:
  roar = a CERTIFIED, TUNABLE comm-vs-removal FRONTIER (τ knob); FedOSD is the expensive
  τ→1 endpoint with no certificate and no choice.
- **"Unlearning-aware CL"** = novel in VFL, exists centralized (2505.15178). Our twist =
  the party-OWNERSHIP-structural objective (concentrate class → fewer future-S* parties).

## ROAR-v2 design (grounded)
- Erase via UCE (FedOSD Eq.3), not GA. [done]
- Orthogonalize erase dir vs retain grad: PCGrad surgery [done] OR GPM null-space (stronger).
- Restrict to S* [done — the novel part; not in surveyed lit].
- Skip recovery (it re-injects); or FedOSD projection-guarded recovery. [done: skip]
- KNOWN LIMIT (confirmed empirically): removal capped by coverage — non-S* evidence survives
  the reconnection attack. roar's comm advantage and removal completeness TRADE OFF (= the
  certificate's (1−τ)Z_f term). Position as a certified frontier, not a fedosd-beater.
- Stronger options if needed: GPM null-space on S*; or SISA×ownership (exact retrain of S*
  encoders from a shard checkpoint) as a gold removal oracle.

## Repositioning implication
Lead the paper on: (A) certified ownership-based minimal-set selection + the comm-vs-leakage
CERTIFIED FRONTIER (τ knob), (B) VFL ownership-concentrating continual learning. Do NOT claim
"better unlearner than fedosd". Benchmark removal under probing/relearning attacks, vs the
fedosd (τ→1) and retrain endpoints.
