# CosineDecayPeak Scheduler Spec

## Goal
Provide a cosine learning-rate scheduler whose **minima are fixed** (`eta_min`) while **cycle peaks evolve over training**:
- default behavior (no warmup): peaks **decay** from `lr_max` to `lr_end` following a polynomial law (matches legacy CosineDecayPeak curve).
- optional warmup: peaks **increase** during warmup from a low value up to `lr_max`, then **decay** afterwards down to `lr_end` (matches the "ramp up then down" envelope).

This scheduler is training-method agnostic (LoRA/LoKR/finetune, etc.) and works for any optimizer.

---

## Definitions

### Global step
- `step`: integer training step index (0-based).
- `max_steps`: total number of training steps (must be > 0).
- progress `p = clamp(step / max_steps, 0, 1)`.

### Cycle structure
- `T_0`: half-cycle length in steps.
- Full cycle length: `cycle_len = 2 * T_0`.
- For each cycle, we generate a cosine wave that goes:
  peak -> eta_min -> peak
  over `cycle_len` steps.

Within a cycle:
- `t = step % cycle_len` (integer)
- `phase = t / T_0`
- For `phase in [0, 1]` we go from peak down to eta_min.
- For `phase in (1, 2]` we go from eta_min back up to peak.

Cosine interpolation factor:
- If `phase <= 1`:
  - `alpha = 0.5 * (1 + cos(pi * phase))`    // 1 -> 0
- Else:
  - `alpha = 0.5 * (1 + cos(pi * (2 - phase)))` // 0 -> 1

Given current cycle peak value `peak_lr`:
- `lr = eta_min + (peak_lr - eta_min) * alpha`

Minima are therefore exactly `eta_min`.

---

## Peak envelope

### Required parameters
- `learning_rate` (aka `lr_max`): target peak learning rate.
- `lr_end` (aka `lr_end`): final peak learning rate at the end of training.
- `eta_min`: minimum LR for the cosine wave (if not explicitly provided, default to `lr_end` or a separate config key; see "Config mapping").
- `lr_power`: polynomial exponent controlling how peaks evolve.
- `T_0`
- `max_steps`

### Peak schedule without warmup (legacy-compatible)
Let `p = step / max_steps` clamped to [0, 1].

Compute polynomial decay factor:
- `d = (1 - p) ** lr_power`

Peak LR is:
- `peak_lr = lr_end + (lr_max - lr_end) * d`

This yields `peak_lr(step=0) = lr_max` and `peak_lr(step=max_steps) = lr_end`.

### Peak schedule with warmup
Add:
- `warmup_steps` (integer, default 0)

We define a **two-phase** envelope:
1) Warmup phase: peaks increase from a start value up to `lr_max`.
2) Decay phase: peaks decrease from `lr_max` down to `lr_end`.

#### Warmup start peak
We need a defined initial peak when warmup is enabled:
- `warmup_start_lr` (optional)
  - default: `lr_end` (or `eta_min` if explicitly set and lower/appropriate)

#### Warmup fraction
- If `warmup_steps > 0`:
  - `pw = clamp(step / warmup_steps, 0, 1)`
- else `pw = 1`.

Warmup peak:
- `peak_warm = warmup_start_lr + (lr_max - warmup_start_lr) * (pw ** lr_power_warmup)`

Where:
- `lr_power_warmup` (optional) defaults to `1.0` (linear warmup),
  but may reuse `lr_power` if a single exponent is preferred.

Decay fraction after warmup:
- Define remaining steps:
  - `remaining = max(max_steps - warmup_steps, 1)`
- If `step <= warmup_steps`: decay progress `pd = 0`
- Else:
  - `pd = clamp((step - warmup_steps) / remaining, 0, 1)`

Decay peak:
- `peak_decay = lr_end + (lr_max - lr_end) * ((1 - pd) ** lr_power)`

Final peak:
- If `step < warmup_steps`:
  - `peak_lr = peak_warm`
- Else:
  - `peak_lr = peak_decay`

This produces the "ramp up then down" envelope (image 2), with warmup defining the rising part.

---

## Clamping and safety rules
- Always enforce:
  - `eta_min <= lr <= peak_lr`
  - `peak_lr >= eta_min`
- If computed `peak_lr < eta_min`, set `peak_lr = eta_min`.
- If user sets inconsistent values (e.g. `lr_end < 0`), raise a config validation error.
- Corner case handling:
  - If `warmup_steps >= max_steps`:
    - scheduler runs only warmup envelope for entire training, and never enters decay.
    - In that case, define `warmup_steps_effective = max_steps`.
    - `pd` remains 0.
- Resume support:
  - scheduler must be fully determined by `step` (no hidden mutable cycle state),
    and must accept a `last_step` / `last_epoch` so resuming yields identical LR.

---

## Config mapping (SimpleTuner-style)
Minimum required keys:
- `--lr_scheduler: cosine_decay_peak`
- `--learning_rate` -> `lr_max`
- `--lr_end` -> `lr_end`
- `--lr_power` -> `lr_power`
- `--lr_scheduler_T_0` (or equivalent existing key) -> `T_0`
- `--max_train_steps` (or computed total steps) -> `max_steps`

New optional keys:
- `--lr_warmup_steps` -> `warmup_steps`
- `--lr_warmup_start` -> `warmup_start_lr` (default `lr_end` or `eta_min`)
- `--lr_warmup_power` -> `lr_power_warmup` (default 1.0)
- `--eta_min` optional if ST uses a dedicated key; otherwise default `eta_min = lr_end`.

---

## Logging expectations
If ST supports LR logging:
- log `lr` each step
- optionally log `peak_lr` (envelope) and `eta_min`
- optionally log `cycle_phase` and `cycle_index`

---

## Acceptance criteria (must pass)
1) **Minima fixed**: at all steps, `lr >= eta_min` and cycle troughs hit `eta_min` (within float tolerance).
2) **Peaks bounded**:
   - Without warmup: `peak_lr` is monotonically non-increasing over training.
   - With warmup: `peak_lr` is non-decreasing for `step < warmup_steps` and non-increasing for `step >= warmup_steps`.
3) **Endpoints**:
   - No warmup: `peak_lr(step=0) == lr_max` and `peak_lr(step=max_steps) == lr_end` (tolerance).
   - With warmup: `peak_lr(step=warmup_steps) == lr_max` (tolerance), and end tends to `lr_end`.
4) **Corner case**: if `warmup_steps >= max_steps`, scheduler does not crash and produces a pure warmup envelope.
5) **Resume**: given identical inputs and `last_step`, LR sequence matches a non-interrupted run.