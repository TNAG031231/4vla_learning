# Stage 3D GRPO Offline Diagnostics

## Finding

The saved step-256 policy has a small **descriptive** improvement over SFT on
3,594 validation observations. Scene-cluster confidence intervals include zero.
The artifacts establish limited observation coverage and real consistency-induced
reward-ranking conflicts; they do not establish either as the sole cause of the
small validation gain. No new model inference, training or reward modification was performed.

## Evidence and intake

Analyzed `grpo_diagnostics_inputs.zip`, supplied by the user, with these members:

- `stage2b_full_run01/{metadata.json,final_validation_predictions.jsonl}`;
- `grpo_pilot_run01/{metadata.json,training_history.jsonl,rollout_1.json,...,rollout_32.json}`;
- `grpo_continuation_to256_run01/{metadata.json,training_history.jsonl,rollout_33.json,...,rollout_256.json}`;
- `grpo_full_validation_reuse_sft_run01/{metadata.json,sft_full_predictions.jsonl,evaluate-full_256_predictions.jsonl}`.

Saved run identities agree across Pilot, continuation and Full Validation:
Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`,
SFT `checkpoint_3564`, identical train-token pool, rollout/reward settings,
monitor population, inference settings and manifest identities. Metadata links
continuation to Pilot `checkpoint_32` and evaluation to continuation `checkpoint_256`.
Training execution commit is `68c2171a711fc8e56355a277c79895159f0f4f42`;
full evaluation is `76e6a475c59f0ebde5726b0d1fb002dc8420e1d0`.
This diagnostic implementation starts from main `eb93536`.

The reused SFT rows exactly equal the original historical SFT prediction file.
SFT/GRPO have identical unique sample populations, scenes, split, action targets,
waypoint targets and masks; all 3,594 predictions are valid on both sides,
covering 140 scenes. Existing `prediction_metrics` recomputation agrees with
saved ADE/FDE within **9.54e-7 m** maximum absolute error (CPU floating-point
rounding). Statistics below use saved per-sample errors. No sampled train scene
or eligible train token overlaps the validation population. No test records or
images were read.

Existing saved temporal-manifest identities agree across GRPO runs:

| Split | Saved SHA-256 |
| --- | --- |
| Train | `98e053f12e494cc10c435d93699a4a65f218bd309158bb3aa094f603bccc9be2` |
| Validation | `767a03fb0c257ab89f4bd01692061d18eb7eb2de04094e3e677fd9d46e796fa1` |

**Evidence limits:** raw temporal manifests are absent, so their saved hashes
cannot be independently recomputed and GT training-action coverage is unavailable.
Checkpoint state/weights are absent; checkpoint lineage is supported by metadata,
not a binary inspection. Historical SFT metadata records temporal schema
`phase0.4-temporal-waypoint-v0.1` and factorized rule `phase0.4-factorized-action-v0.1`;
exact prediction targets agree, but this does not replace raw-manifest verification.
`baseline_source.json` and SFT `training_summary.json` are absent, limiting independent
confirmation of historical decoding provenance and selected SFT checkpoint.
No weights are needed for the statistics delivered here.

## Paired Full Validation

Delta is **GRPO minus SFT** in metres; negative is better. Each statistic uses the
same jointly valid samples with the required horizon. Bootstrap uses seed
`20261116`, 10,000 draws of 140 scenes with replacement, retaining every observation
in each selected scene. Each draw divides summed paired differences by its total
sample count, preserving the sample-weighted estimand and within-scene correlation.
Intervals are percentile 95% CIs. They measure validation scene-sampling uncertainty,
not variation across training seeds. Exact saved-value ties are counted separately.

| Metric | SFT | GRPO | Mean delta | Scene-cluster 95% CI | Improved / tied / worse |
| --- | ---: | ---: | ---: | --- | --- |
| ADE@3s | 0.750680 | 0.747890 | -0.002790 | [-0.011807, +0.006093] | 1504 / 676 / 1414 |
| FDE@3s | 1.641694 | 1.632449 | -0.009245 | [-0.026475, +0.007044] | 1481 / 726 / 1387 |
| 0.7 ADE + 0.3 FDE | 1.017984 | 1.013257 | -0.004727 | [-0.016208, +0.006244] | 1498 / 675 / 1421 |

ADE improves on **41.85%** of all samples (95% CI **38.66–44.86%**);
FDE on **41.21%** (**38.12–44.18%**). Both median deltas are zero. ADE's
5th/95th percentiles are -0.206667/+0.191375 m; FDE's are
-0.480374/+0.448411 m. Small average gains coexist with materially larger,
opposing sample-level changes. These intervals do not demonstrate equivalence
or absence of a possible benefit.

Per-scene mean ADE improves/ties/worsens in **71/10/59** scenes;
FDE in **70/10/60**. The largest ADE scene improvement is
`8383c0b1422b46ad90e2a64ff3e91fcf` (n=27, -0.255522 m); the largest regression is
`8461c1125eaf46c7abaf4ee18e8c4ee6` (n=34, +0.217613 m).
All 140 scene summaries are emitted in `validation.groups.scene`; a single scene
has no between-scene CI. Scene tokens do not identify road/weather/interaction
categories, which are not available in the supplied artifacts.

GT action strata (coarse factorized motion, not semantic turn/lane-change labels):

| Axis / class | Samples | Scenes | ADE delta | FDE delta |
| --- | ---: | ---: | ---: | ---: |
| Longitudinal / accelerate | 679 | 93 | -0.008229 | -0.014796 |
| Longitudinal / decelerate | 686 | 93 | +0.022170 | +0.037485 |
| Longitudinal / keep | 1616 | 125 | -0.010837 | -0.028230 |
| Longitudinal / stop | 613 | 39 | -0.003486 | -0.005343 |
| Lateral / left | 341 | 48 | +0.025623 | +0.037085 |
| Lateral / right | 417 | 55 | +0.000982 | +0.003893 |
| Lateral / straight | 2836 | 140 | -0.006761 | -0.016747 |

Keep/straight (n=1210) improves by -0.011696/-0.032152 m, while
decelerate/straight (n=608) regresses by +0.015196/+0.027908 m.
Stop has 572/613 ties for each metric, explaining much of the low all-sample
improvement fraction. The JSON includes all 10 observed joint classes,
per-class improvement fractions and cluster CIs. These exploratory strata are
not adjusted for multiple comparisons and should not select a new policy by themselves.

## G=4 reward alignment and consistency

Frozen Turn-2 reward is `exp(-E / 1 m) + 0.5 C`, where
`E = 0.7 ADE@3s + 0.3 FDE@3s` and `C` is the fraction of longitudinal/lateral
predictions matching the trajectory-derived action. Turn 1 uses action reward
separately; summing both turns would not represent the Turn-2 optimization target.
Missing 3s endpoints use the existing ADE-only reward rule, while standalone FDE
and the explicitly weighted validation metric require the endpoint.

Recomputed consistency from generated trajectories/actions, reward formulae,
producer `assign_advantages`, saved group statistics and history counters agree.
There are 1,024 candidates, including **3 invalid trajectories**; ranking excludes
these and reports them separately. All 256 groups retain at least two valid
candidates, producing 1,527 within-group valid candidate pairs. A strict reversal
means higher reward accompanies higher error; ties are separate. Spearman uses
average ranks and correlates reward with **negative** error; constant groups are
undefined and excluded from the reported mean with their denominator disclosed.

| Error objective | Trajectory-only reversals | Combined reversals | Combined conflict groups | Mean group Spearman: trajectory / combined |
| --- | --- | --- | --- | --- |
| Weighted E | 0 / 1410 | 140 / 1410 (9.93%) | 85 / 256 | 1.0000 / 0.8250 (240 groups) |
| ADE | 66 / 1410 | 172 / 1410 (12.20%) | 110 / 256 | 0.9367 / 0.7958 (240 groups) |
| FDE | 59 / 1390 | 181 / 1390 (13.02%) | 114 / 256 | 0.9401 / 0.7788 (238 groups) |

Denominators above exclude tied errors; weighted/ADE have 117 both-tied pairs;
FDE also has 20 error-only ties. There are no reward-only ties in these artifacts.
ADE/FDE reversals already exist without consistency because weighted E trades
one metric against the other. In contrast, the 140 weighted-E reversals are
introduced by consistency on these fixed candidates.
For ADE, consistency introduces 123 new pair reversals and resolves 17 existing
ones (66 → 172 total); for FDE it introduces 131 and resolves 9 (59 → 181).

Adding consistency changes the top-reward candidate set in **35/256 groups
(13.67%)**. Mean weighted-E regret of a top combined-reward candidate is
**0.069180 m**, versus zero for trajectory-only (mean across 256 groups;
uniform average across tied maxima). ADE regret is 0.059902 versus 0.006142 m;
FDE regret is 0.121637 versus 0.016480 m. This is an observed candidate-set
counterfactual, not a prediction of retrained-policy performance.

Concrete evidence: `grpo_pilot_run01/rollout_7.json`:

| Candidate | Weighted E (m) | Trajectory reward | C | Combined reward |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3.561550 | 0.028395 | 0.5 | 0.278395 |
| 1 | 6.506781 | 0.001493 | 1.0 | 0.501493 |
| 2 | 14.253305 | 0.000001 | 1.0 | 0.500001 |
| 3 | 9.078805 | 0.000114 | 1.0 | 0.500114 |

The 0.25 consistency bonus exceeds the trajectory-reward difference, favoring
candidate 1 over candidate 0 despite worse weighted error. The exponential
compresses reward differences at large errors; consistency can then dominate
ranking. This is a verified reward trade-off, not proof of generalized reward hacking.

## Training exposure and usable advantage

Pilot contributes steps 1–32 and continuation 33–256: no missing/duplicate steps,
256 G=4 groups and matching history/rollout observation tokens. All **256
observations are unique**, spanning **196 train scenes**; repeated draws = **0**.
Coverage is **256/14,253 = 1.7961%** of the eligible train pool. Four candidates
share one observation and do not multiply observation coverage.

| Turn | Nonzero-variance groups | Zero-variance groups | Groups with nonzero advantage and present tokens |
| --- | --- | --- | --- |
| Action | 162/256 (63.28%) | 94/256 | 162/256 |
| Trajectory | 240/256 (93.75%) | 16/256 | 240/256 |

Effective means nonzero group reward variance under the existing producer,
not guaranteed useful generalization or a parameter update caused by that turn.
Zero advantage removes the surrogate signal for that turn; reference KL remains
part of the optimizer objective. Mean logged KL across all 256 steps is
0.000333, gradient norm 0.434036, LoRA update L2 0.000596; clip fraction is zero.
These logs demonstrate small policy movement, not the cause of limited gains.

Generated candidate-action counts (not GT observation classes):

| Action | Count | Action | Count |
| --- | ---: | --- | ---: |
| keep/straight | 355 | keep/left | 26 |
| keep/right | 88 | decelerate/straight | 211 |
| decelerate/left | 8 | decelerate/right | 24 |
| accelerate/straight | 112 | accelerate/left | 27 |
| accelerate/right | 26 | stop/straight | 143 |
| stop/right | 1 | invalid | 3 |

GT training-action distribution remains unavailable: neither rollout nor history
contains the GT label. Generated actions cannot substitute for GT; recovering it
requires the original train-only temporal manifest joined by sample token.

## Next ablations and learning summary

1. **Reward weight:** prioritize a separately approved controlled comparison of
   consistency weights 0 and 0.5 with the same initial SFT, observation schedule,
   step budget, G and inference protocol. Compare weighted E and standalone
   ADE/FDE, invalid rate, action quality and consistency. Current fixed-candidate
   ranking evidence motivates this test; it does not forecast its outcome.
2. **Coverage:** independently vary the observation budget while holding reward,
   LR and G fixed. Track unique observations, GT class exposure (after obtaining
   the manifest), and nonzero-advantage groups. Low coverage is observed; its
   causal contribution still needs a matched experiment. Avoid changing coverage
   and reward simultaneously.
3. **Uncertainty:** predefine validation endpoints and subgroup reporting before
   the next run. Replicated training seeds are needed to establish optimization
   stability; scene bootstrap alone does not provide it. Keep test data sealed.

The supported engineering capability is **artifact intake → exact paired
alignment → shared metric reproduction → scene-aware uncertainty → per-group
reward decomposition → coverage/advantage accounting**. A defensible interview
explanation is: “I found that the average GRPO gain was smaller than validation
scene-sampling uncertainty, separated weighted-objective trade-offs from
consistency-induced inversions, and measured the actual unique training exposure.
I proposed isolated ablations rather than attributing the result to one cause.”
Follow-up questions include why scene rather than frame bootstrap is needed,
why mean gains can coexist with fewer than half of samples improving, why the
exponential reward changes sensitivity, and why candidate ranking is not a
retraining experiment. This establishes offline diagnosis, not a new training
result, closed-loop performance or deployment capability.

## Reproduction and verification

Executed on CPU in `codex4vla_env`; ZIP members are read directly without
extraction. The CLI is scoped to the supplied Stage 3D export directory names
listed above. Outputs remain outside Git and include every scene/action group,
per-rollout rank statistics, pair transitions, original action metrics and
provenance limitations. No new dependency is required.

Executed command (portable paths shown through shell variables):

```bash
INPUT_ARCHIVE="$HOME/Downloads/grpo_diagnostics_inputs.zip"
OUTPUT_JSON="/tmp/grpo_offline_diagnostics_final.json"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 conda run --no-capture-output -n codex4vla_env \
  python scripts/diagnose_drivema_grpo.py --archive "$INPUT_ARCHIVE" --output "$OUTPUT_JSON"
```

Output must be a new file; choose a new filename for a rerun. The default bootstrap
is 10,000 replicates, seed 20261116. Tests use existing evaluator/reward producers
and cover clustered resampling with unequal scene sizes, ties, invalid outputs,
missing horizons, reward conflicts, repeated observations, external contract
failures and the archive-to-CLI output path. Synthetic test numbers are not
experimental results. Final CPU suite: **164 passed**, including 19 diagnostics
tests, with 13 existing PEFT warnings (50.62 seconds); see the PR verification
record for the full command. Staged diff and workspace hygiene checks passed.
