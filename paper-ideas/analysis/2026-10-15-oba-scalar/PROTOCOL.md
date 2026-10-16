# OBA scalar-module adaptation protocol

Written before running this experiment. This is NOT the full MTSP-LDP algorithm.

- Source: https://arxiv.org/html/2512.24899v1, Section 3.2 and Algorithm 2 / Equation 1.
- No confirmed author implementation found in the title/code searches or author publication page checked on 2026-10-15. This does not establish that no code exists.
- Same data family as round 6: n=42000, 15 pairs, w=20, T=240, 1000 repetitions. This round fixes epsilon=1 and adds a stationary control with three block correlations (0.2,-0.4,0.6).
- Primary constraint for this diagnostic: stationary-control maximum-over-pairs time-averaged MSE <=0.01, using timestamps 40:240. This is an experimental working target, not an application-derived threshold or a user-approved final task definition.
- Among methods meeting that point-estimate constraint, compare changed-pair first-10-round MSE for persistent and 8-round pulse cases. Also report event-active and post-return errors separately. No hyperparameter search.
- Dynamic event indices 80,120,160, amplitudes +0.6,-0.6,+0.6. Only the evaluator knows them.
- Every user is publicly assigned to exactly one pair. The adaptation uses n/15 users per pair each round. Monitoring budget is epsilon/(2w), publication budget uses OBA's selected k and is capped by the actual remaining epsilon/2 publication ledger.
- First w rounds use epsilon/w randomized response, with half recorded in each budget account; history for the OBA score starts at zero. Prewarm transition excluded by the 40-round evaluation warmup.
- Replace the frequency oracle by binary randomized response for the sign X_a X_b. In sign units, the conditional mechanism variance is V(a)=(coth(a/2)^2-1)/(n/15). This is exact for fixed observed records, not the population MSE, which includes sampling variance.
- Score is (monitor_estimate - last_published_estimate)^2 - V(monitor_budget), retaining negative scores. Cost: k*V(epsilon/(2k)) + sum of unselected scores; evaluate all k=0..w. Smallest k wins cost ties; newest timestamp wins score ties.
- Publication accesses the SAME underlying records as monitoring, with fresh randomized-response coins. Sample latent plus counts and then monitor/publication conditional counts. Do not treat the two channels as independent draws of raw records.
- Output is clipped to [-1,1]. No adaptive tree pruning, original grouping/smoothing, or multi-task queries. Additional fixed h5/h20 averages are separately named postprocessing variants; allocation always uses the raw latest publication.
- Communication cap for this NEW comparison is two randomized-response bits per user-round, excluding public control information. Earlier baselines use at most one. All methods share the same two-bit ceiling, but communication usage is not equal. Do not call this an exact continuation of round 6's one-bit cap.
- Baselines: population rotation h1/h5/h20/h40, z2/z3 two-window selection, uniform-budget h20, constant zero. All outputs clipped. Fixed seed 2026101507; confidence intervals from independent simulation runs.
- No result of this reduced module establishes performance of full MTSP-LDP or an advantage over that paper.

## Later audit (2026-10-16; original experiment unchanged)

- Source Section 2.6 explicitly permits population division for MTSP-LDP. This adaptation does not implement it; the rotating baselines therefore also differ in data-collection schedule.
- Source Algorithm 1 / Section 3.2 feeds the grouped/smoothed release into subsequent decisions. Here h5/h20 are final postprocessing only; they do not reproduce that feedback path.
- See the round-8 report for heteroscedasticity and reused-output questions when extending GroupSmooth. These do not invalidate the differently defined monitoring score in this protocol.
