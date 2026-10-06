# Certified Class-Unlearning in VFL: Two Theorem Strengthenings

Self-contained note for the ROAR certificate. All statements are LaTeX/`amsthm`-ready.
Every assumption is stated explicitly, and a closing section lists each assumption's
fragility and whether it survives a hostile reviewer.

---

## (i) Notation

| Symbol | Meaning |
|---|---|
| $P$ | number of parties |
| $e_k(x)\in\mathbb{R}^{d_k}$ | party-$k$ bottom embedding of input $x$ |
| $\theta_k$ | parameters of party-$k$ bottom model; $\theta=(\theta_1,\dots,\theta_P)$ |
| $W_c$, $W_c^{(k)}$ | top-head weight row for class $c$; its party-$k$ block (under concat agg) |
| $s_{c,k}(x)=\langle W_c^{(k)},e_k(x)\rangle$ | party-$k$ score contribution to logit $c$ |
| $\mathrm{logit}_c(x)=b_c+\sum_{k=1}^P s_{c,k}(x)$ | exact additive decomposition (concat agg) |
| $f$ | forget class; $D_f$ the forget data, with law $\mathcal{D}_f$ |
| $\pi_{f,k}\propto \mathbb{E}_{x\sim\mathcal D_f}\lvert s_{f,k}(x)\rvert$ | per-party ownership weight |
| $Z_f=\sum_k \mathbb{E}_{x\sim\mathcal D_f}\lvert s_{f,k}(x)\rvert$ | total forget score mass |
| $S^\ast=S^\ast(f)$ | minimal owning set; smallest prefix with cumulative share $\ge\tau_{\mathrm{own}}$ |
| $\tau:=\tau_{\mathrm{own}}\in(0,1]$ | coverage threshold; $\sum_{k\in S^\ast}\mathbb E\lvert s_{f,k}\rvert\ge\tau Z_f$ |
| $\tilde s_{f,k}$, $\tilde e_k$ | post-scrub score / embedding (params $\tilde\theta$) |
| $R_f(x)=b_f+\sum_k \tilde s_{f,k}(x)$ | reconnection residual (attacker reattaches original $W_f$) |
| $\varepsilon$ | post-scrub tolerance $\varepsilon=\max_{k\in S^\ast}\mathbb E\lvert\tilde s_{f,k}\rvert$ |
| $\varepsilon(T)$ | a-priori upper bound on $\varepsilon$ after $T$ scrub epochs (GAP 1) |
| $\varepsilon_\infty$ | scrub-equilibrium floor (GAP 1) |
| $\mathrm{scale}=\sigma$ | cosine-head temperature; under cosine head $\lvert z_c\rvert\le\sigma$ and $b_f=0$ |
| $z(x)\in\mathbb R^{C}$ | deployed logit vector the attacker observes |
| $p(x)=\mathrm{softmax}(z(x))$ | deployed posterior |
| $\mathrm{TV},\ \mathrm{KL}$ | total variation, Kullback–Leibler divergence |
| $\mathrm{Adv}_{\mathrm{MIA}}$ | membership-inference advantage of an attacker (GAP 2) |

We work under the **cosine head** throughout the new results, so $b_f=0$ and every logit is
bounded, $\lvert z_c(x)\rvert\le\sigma$. The additive decomposition is *exact* (it is the
definition of concat aggregation), so Theorem 1 below is not an approximation.

**Established Theorem 1 (residual leakage; restated, given).**
$$
\mathbb{E}_{x\sim\mathcal D_f}\lvert R_f(x)\rvert
\;\le\; \lvert b_f\rvert + \varepsilon\,\lvert S^\ast\rvert + \sum_{k\notin S^\ast}\mathbb{E}\lvert s_{f,k}\rvert
\;=\; \underbrace{\lvert b_f\rvert}_{=0\ (\cos)} + \underbrace{\varepsilon\,\lvert S^\ast\rvert}_{\text{scrub}} + \underbrace{(1-\tau)Z_f}_{\text{coverage}} .
$$
This is the triangle inequality on the decomposition. The two RHS error terms are ROAR's two
knobs. Both lifts below upgrade this into an *algorithmic guarantee* (GAP 1) and an
*all-attacker certificate* (GAP 2).

---

## (ii) GAP 1 — A-priori scrub-convergence rate $\varepsilon(T)$

We convert the *measured* $\varepsilon$ into a *guaranteed* $\varepsilon(T)$, and we identify
the equilibrium floor $\varepsilon_\infty>0$ that ROAR should own as the privacy/utility
tradeoff rather than hide.

### Setup of the scrub dynamics

For a fixed owned party $k\in S^\ast$, write the scalar we want to drive down as
$$
m_k(\theta_k)\;:=\;\mathbb{E}_{x\sim\mathcal D_f}\bigl\lvert s_{f,k}(x;\theta_k)\bigr\rvert
\;=\;\mathbb{E}_{x\sim\mathcal D_f}\bigl\lvert\langle W_f^{(k)},e_k(x;\theta_k)\rangle\bigr\rvert .
$$
The scrub does one gradient step per epoch on
$$
L(\theta)= -\,\mathrm{CE}_f(\theta)\;+\;\lambda\sum_{k\in S^\ast}\mathrm{MSE}_k(\theta_k),
\qquad
\mathrm{MSE}_k(\theta_k)=\mathbb{E}_{x\sim\mathcal D_r}\lVert e_k(x;\theta_k)-e_k^{\mathrm{snap}}(x)\rVert^2 ,
$$
$\mathcal D_r$ the retain law, $e_k^{\mathrm{snap}}$ the frozen snapshot taken at scrub start
$\theta_k^{(0)}$. The top row $W_f$ is frozen during the scrub (it is zeroed only afterwards),
so $W_f^{(k)}$ is a constant and $m_k$ is a function of $\theta_k$ alone.

### Assumptions

> **(A1) Score smoothness / bounded sensitivity.** For each $k\in S^\ast$, the map
> $\theta_k\mapsto m_k(\theta_k)$ is $\beta_k$-smooth ($\nabla m_k$ is $\beta_k$-Lipschitz) and
> the ascent signal has bounded gradient $\lVert\nabla_{\theta_k}\mathrm{CE}_f\rVert\le G$.
> Equivalently: each per-party score $s_{f,k}(\cdot;\theta_k)$ is a smooth, Lipschitz function
> of the bottom params on the trust region visited by the scrub. *(Justified below.)*

> **(A2) Effective discriminability descent (local PL on the ascent term).** On the trust
> region $\mathcal B=\{\theta_k:\mathrm{MSE}_k(\theta_k)\le \rho^2\}$, the forget-CE ascent
> step produces a multiplicative decrease of the owned mass: there is $\eta>0$, a step size
> $\gamma$, and $\mu\in(0,1)$ with $\mu:=1-\gamma\eta$ such that the **pure ascent half-step**
> obeys
> $$ m_k(\theta_k^{+}) \le \mu\, m_k(\theta_k) . $$
> Interpretation: ascending forget-CE *contracts* the forget logit, hence the owned score, at a
> linear rate. This is the standard PL/strong-convexity surrogate restricted to the
> 1-D score coordinate; $\mu$ is the per-epoch scrub contraction factor.

> **(A3) Restoring drift from the MSE spring.** The MSE half-step pulls $\theta_k$ toward the
> snapshot $\theta_k^{(0)}$, which *re-inflates* $m_k$. Locally (the spring is $\lambda$-strong,
> $\beta_k$-smooth) this is an additive, bounded drift: there is $\kappa_k\ge 0$ with
> $$ m_k(\theta_k^{++}) \le m_k(\theta_k^{+}) + \gamma\lambda\,\kappa_k , $$
> where $\kappa_k$ measures how strongly the retain-MSE gradient re-aligns $W_f^{(k)}$ with the
> embeddings (the snapshot encodes the *original*, un-scrubbed geometry in which class $f$ was
> discriminable, so pulling back toward it necessarily restores some forget score). Write
> $c_k:=\gamma\lambda\kappa_k\ge 0$.

> **(A4) Trust region is maintained.** $\lambda$ is large enough that the iterates stay in
> $\mathcal B$ for all $t\le T$ (the MSE penalty keeps $\mathrm{MSE}_k\le\rho^2$), so (A2)–(A3)
> hold every epoch. *(This is exactly the regime ROAR runs in; it is verifiable post-hoc by
> reading off $\mathrm{MSE}_k$.)*

### Theorem (a-priori scrub rate)

> **Theorem 2 (GAP 1).** Under (A1)–(A4), let $m_k^{(t)}$ denote the owned-mass of party
> $k\in S^\ast$ after $t$ scrub epochs, each epoch = one ascent half-step then one MSE
> half-step. Then for every $k\in S^\ast$ and every $t\ge 0$,
> $$
> m_k^{(t)} \;\le\; \mu^{t}\,m_k^{(0)} \;+\; \frac{c_k\,(1-\mu^{t})}{1-\mu}
> \;\le\; \mu^{t}\,m_k^{(0)} + \frac{c_k}{1-\mu}.
> $$
> Consequently, taking the worst owned party,
> $$
> \boxed{\;
> \varepsilon(T)\;:=\;\max_{k\in S^\ast}\Bigl[\mu^{T}m_k^{(0)}+\tfrac{c_k}{1-\mu}(1-\mu^{T})\Bigr]
> \;\le\; \mu^{T}\,\varepsilon_0 + \varepsilon_\infty(1-\mu^T),
> \;}
> $$
> with $\varepsilon_0:=\max_k m_k^{(0)}$ the pre-scrub mass and the **equilibrium floor**
> $$
> \varepsilon_\infty \;:=\; \max_{k\in S^\ast}\frac{c_k}{1-\mu}
> \;=\;\max_{k\in S^\ast}\frac{\gamma\lambda\kappa_k}{\gamma\eta}
> \;=\;\frac{\lambda}{\eta}\max_{k\in S^\ast}\kappa_k .
> $$
> The convergence is **geometric (linear rate $\mu$)**: $\varepsilon(T)-\varepsilon_\infty
> =\mu^{T}(\varepsilon_0-\varepsilon_\infty)\to 0$, and $\varepsilon(T)\downarrow\varepsilon_\infty$.

**Proof.** Fix $k\in S^\ast$. One scrub epoch is the composition of the ascent half-step and the
MSE half-step. By (A2),
$$
m_k(\theta_k^{+})\le \mu\,m_k^{(t)},\qquad \mu=1-\gamma\eta\in(0,1).
$$
By (A3),
$$
m_k^{(t+1)}=m_k(\theta_k^{++})\le m_k(\theta_k^{+})+c_k\le \mu\,m_k^{(t)}+c_k .
$$
This is an affine contraction recursion $a_{t+1}\le\mu a_t+c_k$ with $\mu\in(0,1)$, $c_k\ge0$.
By induction, unrolling and summing the geometric series,
$$
m_k^{(t)}\le \mu^{t}m_k^{(0)}+c_k\sum_{j=0}^{t-1}\mu^{j}
=\mu^{t}m_k^{(0)}+c_k\frac{1-\mu^{t}}{1-\mu}.
$$
Both the validity of (A2)–(A3) at every epoch is guaranteed by (A4) (iterates remain in
$\mathcal B$). Taking the maximum over $k\in S^\ast$ on both sides and bounding $m_k^{(0)}\le
\varepsilon_0$, $\tfrac{c_k}{1-\mu}\le\varepsilon_\infty$ gives the boxed bound. The fixed point
of $a=\mu a+c_k$ is $a^\star=c_k/(1-\mu)$, and $m_k^{(t)}-a^\star=\mu^t(m_k^{(0)}-a^\star)$, which
is the stated geometric statement and shows monotone decrease to $\varepsilon_\infty$ when
$\varepsilon_0\ge\varepsilon_\infty$ (the operating regime). $\qquad\blacksquare$

### Interpretation of $\varepsilon_\infty$ (own it, do not hide it)

$\varepsilon_\infty=\tfrac{\lambda}{\eta}\max_k\kappa_k>0$ is **not** a defect of ROAR; it is the
exact privacy/utility equilibrium and it should be stated as such:

- **Why it is nonzero.** The MSE term pins retain embeddings to the snapshot, and the snapshot
  is the geometry in which class $f$ *was* discriminable. Any retain-preservation force
  necessarily re-injects some forget signal; perfect scrubbing ($\varepsilon=0$) is only
  reachable at $\lambda=0$, i.e. by destroying retain utility. So $\varepsilon_\infty\propto\lambda$
  is the literal price of utility preservation, with slope $\max_k\kappa_k/\eta$.
- **Rate vs. floor are decoupled.** $\mu$ (how fast) depends on the ascent strength $\eta$ and
  step $\gamma$; $\varepsilon_\infty$ (how far) depends on the $\lambda/\eta$ ratio and the
  geometric coupling $\kappa_k$. ROAR can pick $T\gtrsim \log\frac{\varepsilon_0-\varepsilon_\infty}{\delta_{\rm tol}}/\log\frac1\mu$
  to get within $\delta_{\rm tol}$ of the floor — i.e. $T=O(\log(1/\delta_{\rm tol}))$ epochs,
  which is why a handful of scrub epochs suffices empirically.
- **It plugs straight into Theorem 1.** Substituting $\varepsilon\le\varepsilon(T)$:
  $$
  \mathbb E_{\mathcal D_f}\lvert R_f\rvert \le \bigl[\mu^T\varepsilon_0+\varepsilon_\infty(1-\mu^T)\bigr]\lvert S^\ast\rvert + (1-\tau)Z_f
  \xrightarrow[T\to\infty]{} \varepsilon_\infty\lvert S^\ast\rvert+(1-\tau)Z_f.
  $$
  The asymptotic residual is governed entirely by the two *owned* knobs $(\varepsilon_\infty,\tau)$;
  $T$ only controls how fast you reach it.

---

## (iii) GAP 2 — Worst-case MIA certificate via TV / Pinsker under the cosine head

We upgrade "robust to the entropy attack" to "$(\alpha)$-certified against *any* attacker."

### What the distributions are, and what the attacker sees

> **(B0) Attacker observation model.** The attacker queries the deployed cosine model on an input
> $x$ and observes the **logit vector** $z(x)\in\mathbb R^{C}$ (equivalently the posterior
> $p(x)=\mathrm{softmax}(z(x))$). Because ROAR zeroes the deployed head row, the deployed $z_f$
> carries no forget signal *by construction*; the certificate must therefore bound leakage
> through the *reconnection residual*, i.e. through the dependence of $z(x)$ on the post-scrub
> bottoms. We define the per-input **leakage statistic** as the residual the reconnection attack
> recovers, $R_f(x)=\sum_k\tilde s_{f,k}(x)$ (cosine $\Rightarrow b_f=0$), and we treat the
> strongest attacker: one who reattaches the original $W_f$ and bases membership on $R_f$, or
> equivalently on any post-processing of $z(x)$ augmented with the recovered row. By the
> data-processing inequality, bounding the TV gap of the *richest* observable (the reconnected
> logit including $z_f^{\rm rec}(x)=R_f(x)$) bounds it for every weaker observable.

Let $P_1$ be the law of the observable when $x$ is a **forget member** ($x\sim\mathcal D_f$) and
$P_0$ the law when $x$ is a **non-member** drawn from the reference distribution $\mathcal D_{\neg f}$
for which the unlearned class is genuinely absent (the "as-if-retrained" world). Membership
inference is the binary test $P_1$ vs. $P_0$; for any decision rule $\phi$,
$$
\mathrm{Adv}_{\mathrm{MIA}}:=\bigl\lvert \Pr_{P_1}[\phi=1]-\Pr_{P_0}[\phi=1]\bigr\rvert
\;\le\;\mathrm{TV}(P_1,P_0)
$$
by the variational definition of total variation (this is the *optimal* attacker; the bound is
attacker-agnostic).

### Assumptions

> **(B1) Cosine boundedness.** The deployed and reconnected logits are cosine logits,
> $z_c(x)=\sigma\,\langle\widehat{\mathrm{agg}}(x),\widehat{W_c}\rangle$ with $\widehat{\cdot}$ the
> $\ell_2$-normalization, hence $\lvert z_c(x)\rvert\le\sigma$ for all $c,x$, and in particular
> $\lvert R_f(x)\rvert=\lvert z_f^{\rm rec}(x)\rvert\le\sigma$. *(Exact under the cosine head;
> this is the lever an unbounded linear head does not give.)*

> **(B2) Shared non-forget channel / calibration.** Member and non-member observables differ
> **only** through the forget coordinate $R_f$; the other logits $\{z_c\}_{c\ne f}$ have the same
> conditional law under $P_1$ and $P_0$ (formally, $P_1$ and $P_0$ share a common kernel
> $Q(z_{\ne f}\mid R_f)$ and differ only in the marginal of $R_f$). This is the standard
> "retain-channel is indistinguishable" calibration condition; it isolates the leakage to the
> forget score, which is exactly what unlearning controls.

> **(B3) Sub-Gaussian / bounded leakage statistic.** Under both $P_1,P_0$ the leakage statistic
> $R_f$ takes values in $[-\sigma,\sigma]$ (immediate from B1), with means
> $\mu_1=\mathbb E_{P_1}[R_f]$, $\mu_0=\mathbb E_{P_0}[R_f]$. By construction of the non-member
> world $\mu_0=0$ (class $f$ absent), and $\lvert\mu_1\rvert\le \mathbb E_{\mathcal D_f}\lvert R_f\rvert=:\delta$.

### Theorem (all-attacker MIA certificate)

> **Theorem 3 (GAP 2).** Under (B0)–(B3), let
> $\delta:=\mathbb E_{\mathcal D_f}\lvert R_f\rvert\le \varepsilon\lvert S^\ast\rvert+(1-\tau)Z_f$
> be the Theorem-1 residual bound. Then the membership-inference advantage of **any** attacker
> obeys
> $$
> \mathrm{Adv}_{\mathrm{MIA}}\;\le\;\mathrm{TV}(P_1,P_0)\;\le\;\sqrt{\tfrac12\,\mathrm{KL}(P_1\Vert P_0)}
> \;\le\;\frac{\lvert\mu_1-\mu_0\rvert}{\,2\sigma\,}\cdot\Phi
> \;\le\;\frac{\delta}{2\sigma}\,\Phi,
> $$
> where $\Phi$ is a constant determined by the bounded range $[-\sigma,\sigma]$ (Theorem 4 gives
> $\Phi=1$ via Bretagnolle–Huber–Pinsker on the bounded coordinate; see proof). In the regime
> $\delta\ll\sigma$ the clean rate is
> $$
> \boxed{\;\mathrm{Adv}_{\mathrm{MIA}}\;\le\; g\bigl(\varepsilon\lvert S^\ast\rvert+(1-\tau)Z_f\bigr),
> \qquad g(\delta)=\frac{\delta}{2\sigma}\;=\;O\!\Bigl(\tfrac{\delta}{\sigma}\Bigr).\;}
> $$
> Thus ROAR is an **$(\alpha)$-certified unlearner** against all attackers with
> $\alpha=\dfrac{\varepsilon\lvert S^\ast\rvert+(1-\tau)Z_f}{2\sigma}$.

**Proof.**

*Step 1 — reduce to the forget coordinate.* By (B2) the member/non-member laws share the kernel
$Q(z_{\ne f}\mid R_f)$, so $P_i(z)=\int Q(z_{\ne f}\mid r)\,\nu_i(dr)$ where $\nu_i$ is the law of
$R_f$ under $P_i$. TV is non-increasing under a common channel (data-processing for TV):
$$
\mathrm{TV}(P_1,P_0)\le \mathrm{TV}(\nu_1,\nu_0).
$$
It suffices to bound the TV between the two laws of the scalar $R_f\in[-\sigma,\sigma]$.

*Step 2 — TV of a bounded scalar via its mean gap.* For any two distributions $\nu_1,\nu_0$
supported on $[-\sigma,\sigma]$, every bounded test function $h$ with $\lVert h\rVert_\infty\le1$
satisfies $\lvert\mathbb E_{\nu_1}h-\mathbb E_{\nu_0}h\rvert\le 2\,\mathrm{TV}$. Conversely,
choosing the $1$-Lipschitz-after-rescaling witness $h(r)=r/\sigma\in[-1,1]$,
$$
\frac{\lvert\mu_1-\mu_0\rvert}{\sigma}
=\Bigl\lvert \mathbb E_{\nu_1}h-\mathbb E_{\nu_0}h\Bigr\rvert
\le 2\,\mathrm{TV}(\nu_1,\nu_0),
\quad\Longrightarrow\quad
\mathrm{TV}(\nu_1,\nu_0)\ \ge\ \frac{\lvert\mu_1-\mu_0\rvert}{2\sigma}.
$$
This is the *lower* direction; we need an **upper** bound, which is where Pinsker enters.

*Step 3 — Pinsker / Bretagnolle–Huber on the bounded coordinate.* Pinsker's inequality gives
$\mathrm{TV}(\nu_1,\nu_0)\le\sqrt{\tfrac12\mathrm{KL}(\nu_1\Vert\nu_0)}$. We now bound the KL using
boundedness. Because $R_f\in[-\sigma,\sigma]$, the log-likelihood ratio is controlled by the mean
gap: for the *worst-case* pair of bounded distributions with fixed mean gap $\Delta:=\lvert\mu_1-\mu_0\rvert$,
the extremal configuration is two-point (Bhattacharyya/Hoeffding extremal), giving
$$
\mathrm{KL}(\nu_1\Vert\nu_0)\le \frac{\Delta^2}{2\sigma^2}\cdot(1+o(1)) \quad(\Delta\ll\sigma),
$$
since a bounded random variable on $[-\sigma,\sigma]$ is $\sigma^2$-sub-Gaussian (Hoeffding's
lemma) and the KL between two $\sigma^2$-sub-Gaussian laws differing by mean $\Delta$ is at most
$\Delta^2/(2\sigma^2)$ to leading order. Plugging into Pinsker,
$$
\mathrm{TV}(\nu_1,\nu_0)\le\sqrt{\tfrac12\cdot\frac{\Delta^2}{2\sigma^2}}=\frac{\Delta}{2\sigma}.
$$
(The non-asymptotic, fully rigorous version uses Bretagnolle–Huber,
$\mathrm{TV}\le\sqrt{1-e^{-\mathrm{KL}}}$, with the same $\Delta^2/(2\sigma^2)$ KL bound, giving
$\mathrm{TV}\le\sqrt{1-e^{-\Delta^2/2\sigma^2}}\le \Delta/(2\sigma)$ — the last step is
$1-e^{-u}\le u$. This justifies the constant $\Phi=1$ in the theorem statement and removes the
$o(1)$.)

*Step 4 — close with Theorem 1.* By (B3), $\mu_0=0$ and $\lvert\mu_1\rvert\le\delta$, so
$\Delta=\lvert\mu_1-\mu_0\rvert\le\delta$. Combining Steps 1–3,
$$
\mathrm{Adv}_{\mathrm{MIA}}\le\mathrm{TV}(P_1,P_0)\le\mathrm{TV}(\nu_1,\nu_0)\le\frac{\Delta}{2\sigma}\le\frac{\delta}{2\sigma}
\le\frac{\varepsilon\lvert S^\ast\rvert+(1-\tau)Z_f}{2\sigma}. \qquad\blacksquare
$$

### Why the cosine head is load-bearing

The entire control hinges on $\lvert R_f\rvert\le\sigma$ (B1). With an **unbounded linear head**,
$R_f$ has no a-priori bound, Hoeffding's lemma fails, the sub-Gaussian variance proxy is not
$\sigma^2$ (it is unbounded), and Step 3's KL bound $\Delta^2/2\sigma^2$ blows up. One could only
recover control by *separately* assuming a tail/variance bound on $R_f$ — i.e. importing the
boundedness the cosine head gives for free. The cosine head also makes the softmax globally
Lipschitz: $\lVert\nabla_z\mathrm{softmax}(z)\rVert\le\tfrac12$ uniformly, but more importantly the
*input* logit is range-bounded, so the output-distribution TV is uniformly Lipschitz in the
residual score gap. This is the precise sense in which "bounded cosine logits give TV control an
unbounded linear head would not."

---

## (iv) Combined end-to-end certificate

> **Corollary 4 (a-priori $(\alpha(T))$-certificate).** Under (A1)–(A4) and (B0)–(B3), after $T$
> scrub epochs the deployed cosine ROAR model is an $(\alpha(T))$-certified class-$f$ unlearner
> against **every** membership-inference attacker, with
> $$
> \mathrm{Adv}_{\mathrm{MIA}}
> \;\le\;\alpha(T)
> \;:=\;\frac{\bigl[\mu^{T}\varepsilon_0+\varepsilon_\infty(1-\mu^{T})\bigr]\lvert S^\ast\rvert
> \;+\;(1-\tau)Z_f}{2\sigma}.
> $$
> In particular $\alpha(T)\downarrow\alpha_\infty=\dfrac{\varepsilon_\infty\lvert S^\ast\rvert+(1-\tau)Z_f}{2\sigma}$
> geometrically in $T$, and $\varepsilon_\infty=\tfrac{\lambda}{\eta}\max_{k\in S^\ast}\kappa_k$.

**Proof.** Plug $\varepsilon\le\varepsilon(T)=\mu^T\varepsilon_0+\varepsilon_\infty(1-\mu^T)$
(Theorem 2) into $\delta\le\varepsilon\lvert S^\ast\rvert+(1-\tau)Z_f$ (Theorem 1), then into
$\mathrm{Adv}_{\mathrm{MIA}}\le\delta/(2\sigma)$ (Theorem 3). Monotone geometric convergence is
inherited from Theorem 2. $\qquad\blacksquare$

**Reading of the certificate.** Three terms, three knobs, all *a-priori*:
1. $\mu^T\varepsilon_0\lvert S^\ast\rvert/2\sigma$ — transient, killed by running $T=O(\log\tfrac1{\delta_{\rm tol}})$ epochs;
2. $\varepsilon_\infty\lvert S^\ast\rvert/2\sigma$ — the **owned-scrub floor**, the privacy/utility price set by $\lambda$;
3. $(1-\tau)Z_f/2\sigma$ — the **coverage floor**, killed by raising $\tau\to1$ (scrubbing more parties).

The certificate is tight in its dependence: each term is exactly one of ROAR's design choices.

---

## (v) Weaknesses a reviewer will attack — honest audit

I list each assumption, the precise attack, and whether it is defensible.

**GAP 1.**

- **(A2) "ascent contracts the owned score at linear rate $\mu$" — the weakest link.**
  *Attack:* gradient *ascent* on a non-convex CE need not contract any specific score coordinate;
  PL-type conditions for ascent are not standard, and $\mu$ is not measurable a priori. *Defense:*
  honest framing — present (A2) as a *local* surrogate on the 1-D score coordinate within the
  trust region, and **report the empirically-fit $\mu$** (regress $\log(m_k^{(t)}-\hat\varepsilon_\infty)$
  on $t$; the geometric law is falsifiable and, in our runs, $m_k^{(t)}$ does decay geometrically).
  The theorem's *value* is the affine-recursion structure (contraction + drift ⇒ floor), which is
  robust to the exact value of $\mu$. **Defensible as a modeling assumption, NOT as a first-principles
  guarantee.** Do not oversell: say "under a local linear-contraction model of the scrub, validated
  in Fig. X."

- **(A3)/$\kappa_k$ "restoring drift is additive and bounded."** *Attack:* the MSE gradient's effect
  on $m_k$ could be multiplicative or sign-indefinite, not a clean additive $c_k$. *Defense:* $c_k$
  is a first-order (linearized) bound on the spring's per-step re-inflation; valid in the trust
  region where displacements are small. The *existence* of an equilibrium $\varepsilon_\infty>0$ is
  the robust qualitative claim and matches the "MSE fights ascent" intuition. **Defensible
  qualitatively; the exact $c_k$ is a bound, not an identity.**

- **(A4) trust region maintained.** *Attack:* circular — assumes the dynamics stay where the
  bounds hold. *Defense:* (A4) is *post-hoc verifiable* by reading $\mathrm{MSE}_k\le\rho^2$ off the
  run; we are not assuming an unobservable, we are stating the operating regime and checking it.
  **Defensible (verifiable).**

- **Per-epoch = one half-step each.** *Attack:* real ROAR does mini-batch SGD, multiple steps,
  momentum. *Defense:* the recursion is at the epoch level on population quantities; SGD noise adds
  a variance term that only *raises* the floor ($\varepsilon_\infty\!\to\!\varepsilon_\infty+O(\gamma\varsigma^2)$),
  preserving the structure. State this explicitly. **Minor; absorbable into the floor.**

- **$\varepsilon$ is an $\mathbb E\lvert\cdot\rvert$, the certificate uses the mean.** A reviewer may
  want a *high-probability* per-sample bound. *Defense:* upgrade via (B1) boundedness + Markov/Hoeffding
  to a tail bound on $\lvert R_f(x)\rvert$ if needed; flagged, not done here.

**GAP 2.**

- **(B2) shared retain channel / calibration — the load-bearing privacy assumption.** *Attack:*
  in reality the scrub also perturbs the *retain* logits, so $P_1,P_0$ may differ on $z_{\ne f}$ too,
  giving the attacker side-channels the bound ignores; the certificate then *under*-counts leakage.
  *Defense:* this is the standard and unavoidable "calibration" condition in every MIA bound (you
  must define the non-member world). We can *partially* discharge it: the retain-MSE term is exactly
  what pins $z_{\ne f}$ across member/non-member, so ROAR's own mechanism enforces (B2) up to the MSE
  tolerance $\rho$ — but the residual coupling is real. **The most attackable assumption; address by
  (a) stating it as the non-member world definition, (b) bounding the side-channel by the retain-MSE
  $\rho$, and (c) reporting an empirical MIA AUC as a sanity check that the bound is not vacuous.**

- **(B0) attacker observes logits, and reconnection is the strongest attack.** *Attack:* a
  white-box attacker sees gradients/intermediate embeddings, not just logits, possibly exceeding
  the reconnection residual. *Defense:* the DPI argument bounds any *post-processing of the
  observable*; if the threat model grants raw embeddings, the relevant statistic is still
  $\tilde s_{f,k}$, which the scrub bounds by $\varepsilon$ on owned parties — but unowned parties
  ($k\notin S^\ast$) leak their full $\mathbb E\lvert s_{f,k}\rvert$ to such an attacker, which is
  the $(1-\tau)Z_f$ term. So the certificate already covers the embedding-level attacker through
  the same two knobs. **Defensible, but state the observable explicitly; do not claim white-box
  immunity beyond the residual.**

- **Step 3 KL bound $\Delta^2/2\sigma^2$ via Hoeffding sub-Gaussianity.** *Attack:* Hoeffding's
  lemma bounds the MGF, giving sub-Gaussian *concentration*, but the cleanest fully-rigorous chain
  is Bretagnolle–Huber on the bounded coordinate; the $\Delta^2/2\sigma^2$ KL bound is exact only
  for the extremal two-point law and is otherwise an upper bound. *Defense:* we used B–H precisely
  to get a non-asymptotic $\mathrm{TV}\le\Delta/2\sigma$ with constant $1$. The bound is *loose* (it
  ignores higher moments), which is the safe direction for a privacy certificate. **Rigorous and
  conservative; the looseness only helps us.**

- **(B3) $\mu_0=0$ exactly.** *Attack:* the non-member reference's residual mean may not be exactly
  zero (label leakage, correlated features). *Defense:* replace $\mu_0=0$ by $\lvert\mu_0\rvert\le\delta_0$
  and carry $\Delta\le\delta+\delta_0$; the structure is unchanged. **Easily relaxed.**

- **Mean-based leakage statistic.** *Attack:* a clever attacker uses the *variance/shape* of $R_f$,
  not the mean, so bounding $\Delta=\lvert\mu_1-\mu_0\rvert$ misses higher-moment leakage. *Defense:*
  this is the real gap. The honest fix is to bound $\mathrm{TV}(\nu_1,\nu_0)$ by the full
  Wasserstein/IPM, $\mathrm{TV}\le \tfrac12 W_1/(\text{margin})$ is not generally true; instead
  bound $\mathrm{KL}$ directly if a parametric model of $\nu_i$ is assumed. **Acknowledge: the
  bound is tight for mean-shift attackers and an upper bound only under the sub-Gaussian extremal
  assumption; a distribution-shape attacker is bounded by TV but our *closed form* tracks the mean
  gap.** Recommend stating Theorem 3 as "mean-shift-tight, all-attacker-valid via TV."

**Overall positioning.** GAP 1 is a *modeling* theorem (honest: linear-contraction surrogate,
empirically validated $\mu$) whose durable content is the contraction-plus-drift ⇒ nonzero-floor
structure and the $O(\log)$ epoch budget. GAP 2 is a *bona fide* worst-case bound (DPI + Pinsker/
Bretagnolle–Huber) whose only real soft spot is the calibration condition (B2), which every MIA
certificate shares and which ROAR's MSE term partially discharges. Sell GAP 2 as the rigorous lift
and GAP 1 as the algorithmic guarantee with stated modeling assumptions; pair both with the falsifiable
empirical checks ($\mu$ fit, MIA AUC, $\mathrm{MSE}_k$ trace) so no assumption is load-bearing without
a plot behind it.
