# Reconciling GAP1 with the measured U-shaped ε(w_raw)

Status: addresses eval grade-3 blocker B2 (the certificate→action ε(w_raw) curve is
non-monotonic and contradicts GAP1's monotone "geometric convergence to a floor").
Decision: **keep Theorem 1; RETRACT GAP1's global claim, demote to a local guarantee;
explain the U-shape with a derived two-term surrogate (Proposition 2, labelled a
model under assumptions, not an a-priori theorem).**

## The firewall
Theorem 1 (a-posteriori head-reconnection bound) E|R_f| ≤ |b_f| + ε|S*| + (1−τ)Z_f
holds for ANY measured ε at ANY operating point (triangle inequality). The certificate
we ship is "at w_raw=w*, measured ε=1.1 ⇒ E|R_f| ≤ |b_f| + 1.1|S*| + (1−τ)Z_f."
**It does not depend on GAP1 at all.** The U-shape only means the measured ε plugged
into the bound is itself U-shaped; each point is honest.

## Two suppression mechanisms (the combined picture)
The scrub minimizes L = −CE_forget + λ_pres Σ MSE(retain) + w_raw Σ <W_f^(k),e_k>².
- **λ_pres** pins the retain embedding — ISOTROPIC leash; sets a residual floor
  ε_∞ ∝ (λ_pres/η)·max κ_k (GAP1's original term — correct for THIS piece).
- **w_raw** pushes the forget embedding off the rank-one direction W_f^(k) — ANISOTROPIC,
  directly attacks the certified quantity (the knob GAP1 lacked).
They act on the same S* bottoms in non-orthogonal directions → they conflict.

## GAP1 retracted globally → local (monotonicity trap)
GAP1 posited an affine per-epoch recursion m_{t+1} ≤ μ m_t + c (μ∈[0,1), c≥0).
**Lemma (monotonicity trap):** m_t − ε_∞ = μ^t(m_0 − ε_∞); the sequence is MONOTONE and
converges to ε_∞ = c/(1−μ). Holding T fixed and varying w_raw, GAP1 predicts ε monotone
DECREASING in w_raw — it structurally CANNOT rebound. A model that cannot exhibit the
observed behaviour is misspecified outside its regime (not "approximately right").

**Theorem (GAP1, scoped):** on the suppression-binding regime 0 ≤ w_raw ≤ w*, where
(i) μ(w_raw)∈[μ_min,1) is non-increasing in w_raw and (ii) recovery re-injects negligible
forget score (assumption R), the affine recursion holds and
  ε(T;w_raw) = μ(w_raw)^T ε_0 + ε_∞(1−μ(w_raw)^T),  ε_∞ = (λ_pres/η)max κ_k,
monotone decreasing in T and in w_raw. This is why ε plummets 7.4→1.1 from w_raw=0→1.
We forbid extrapolating past w* where (i)/(R) fail.

## Proposition 2 (U-shape surrogate, under M1–M4)
Under linearized embedding response (M1), conflict/leash-violation re-injection (M2),
conditioning degradation κ(w)=1+w/ρ (M3), additive separability (M4):
  ε̄(w) = A·g(w) + B·h(w),  g↓ convex (direct suppression), h↑ (conflict),
has a UNIQUE interior minimizer w* given by A|g'(w*)| = B h'(w*)
(marginal suppression gain = marginal conflict cost). Ridge/linear instance
g(w)=1/(1+w/ρ), h(w)=w gives ε̄(w)=A/(1+w/ρ)+Bw with closed form
  **w* = ρ(√(A/(Bρ)) − 1),  ε̄(w*) = 2√(ABρ) − Bρ > 0** (cannot reach 0).
Data (0,7.4),(1,1.1),(5,1.4),(20,1.7),(50,2.5): A≈7.4, small ρ (steep early drop),
gentle B (slow climb), w*≈1. The nonzero floor matches "E|R_f| never →0" (best ≈2.3).
Reported as shape-consistency of a labelled model, NOT a fitted theorem.

## Mechanism of the rebound (paper body)
Small w_raw: removes the near-orthogonal rank-one forget projection cheaply → ε collapses.
Large w_raw: over-suppresses → must distort the embedding, violating retain-MSE → the
RECOVERY phase (re-fits on retain only, retain shares owned features with the forget class
in Covertype's hardest class) partially UNDOES the suppression, by an amount growing with
w_raw → the B·h(w) term. Secondary route: large w_raw ill-conditions the scrub objective so
fixed-T SGD lands worse (μ degrades). Floored E|R_f| evidence favours re-injection.

## Combined λ_pres × w_raw floor + falsifiable prediction
ε_floor ≈ (λ_pres/η)max κ_k · g(w_raw)  +  B(λ_pres)·h(w_raw).
Conflict amplitude B increases with λ_pres ⇒ **w* shifts LEFT and ε̄(w*) rises as λ_pres
grows** — the discriminating experiment (λ_pres sweep). Discriminate re-injection vs
conditioning by varying recovery epochs (re-injection is recovery-budget sensitive,
conditioning is scrub-T sensitive).

## Honest weaknesses (a theory reviewer will still attack)
1. Prop 2 constants A,B,ρ,g,h are phenomenological (M1–M4), not derived from the Hessian/
   recovery fixed point — "fit a U with a U-shaped family." Defense: claim only
   existence/uniqueness of interior optimum (sign conditions) + structural ε̄(w*)>0 +
   comparative static; labelled Proposition, never Theorem.
2. Recovery phase (the crux/re-injection) is a black box producing B·h(w); no a-priori
   bound on B. Open frontier — flagged.
3. Assumption R is exactly what fails at large w; using it to carve GAP1's valid regime
   then its failure to explain rebound is the correct logical structure (GAP1 true where R
   holds, silent where R fails; w* = where R's violation becomes first-order).
4. g convex / h linear are leading-order; claims restricted to measured range (w≤50).
5. One class / one dataset / 3 seeds; Prop 2 is a transferable mechanism, λ_pres-sweep is
   the committed discriminating test.

## Bottom line
Do not defend GAP1 as stated. Keep the certificate (Thm 1, GAP1-independent); retract the
global monotone-floor claim → local guarantee for w_raw≤w*; explain the rebound with a
derived, clearly-labelled two-term surrogate predicting the U-shape, interior optimum, and
nonzero floor; report "leakage reduced ~3× at the optimal weight" (not "collapses").
A correct narrower claim + a labelled mechanism beats a wrong general theorem.
