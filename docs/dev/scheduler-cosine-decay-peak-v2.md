# CosineDecayPeak Scheduler Spec

## Goal
Provide a cosine learning-rate scheduler whose **upper and lower envelopes both evolve over training**.

Behavior:
- default behavior (no warmup): both the **peak LR** and the **minimum LR** decay polynomially from their initial values toward `lr_end`.
- optional warmup: both envelopes **increase** during warmup up to their configured maxima, then **decay** afterwards down to `lr_end`.
- the cosine wave oscillates between the current upper envelope (`peak_lr`) and the current lower envelope (`floor_lr`).

This scheduler is training-method agnostic (LoRA/LoKR/finetune, etc.) and works for any optimizer.

The intent is to replace the previous "fixed minima" behavior with a more flexible schedule where the cosine range itself evolves over time.

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
  peak -> floor -> peak
  over `cycle_len` steps.

Within a cycle:
- `t = step % cycle_len` (integer)
- `phase = t / T_0`
- For `phase in [0, 1]` we go from peak down to floor.
- For `phase in (1, 2]` we go from floor back up to peak.

Cosine interpolation factor:
- If `phase <= 1`:
  - `alpha = 0.5 * (1 + cos(pi * phase))`    // 1 -> 0
- Else:
  - `alpha = 0.5 * (1 + cos(pi * (2 - phase)))` // 0 -> 1

Given current cycle envelopes `peak_lr` and `floor_lr`:
- `lr = floor_lr + (peak_lr - floor_lr) * alpha`

Therefore:
- `alpha = 1` -> `lr = peak_lr`
- `alpha = 0` -> `lr = floor_lr`

---

## Envelope parameters

### Required parameters
- `learning_rate` (aka `lr_max`): target maximum peak LR.
- `lr_end`: final LR at the end of training; both envelopes converge to this value.
- `lr_power`: polynomial exponent controlling envelope evolution.
- `T_0`
- `max_steps`

### Optional parameters
- `lr_warmup_steps` (aka `warmup_steps`): number of warmup steps for both envelopes.
- `lr_warmup_start` (aka `warmup_start_lr`): starting value for the peak envelope during warmup.
- `lr_floor` (aka `floor_lr_max`): maximum value reached by the lower envelope after warmup, before decay begins.
- `lr_floor_start` (aka `floor_lr_start`): starting value for the lower envelope during warmup.
- `eta_min`: deprecated legacy alias for the lower-envelope maximum in the no-warmup case or for backward compatibility only.

### Envelope semantics
- `peak_lr(step)` is the upper envelope.
- `floor_lr(step)` is the lower envelope.
- At the end of training, both should converge to `lr_end`.

Always enforce:
- `lr_end <= floor_lr(step) <= peak_lr(step)`
- `lr_end <= lr(step) <= peak_lr(step)`

If necessary, clamp `floor_lr(step)` so it never exceeds `peak_lr(step)`.

---

## Peak envelope

### Peak schedule without warmup
Let `p = step / max_steps` clamped to [0, 1].

Compute polynomial decay factor:
- `d = (1 - p) ** lr_power`

Peak LR is:
- `peak_lr = lr_end + (lr_max - lr_end) * d`

This yields:
- `peak_lr(step=0) = lr_max`
- `peak_lr(step=max_steps) = lr_end`

### Peak schedule with warmup
We define a two-phase envelope:
1) Warmup phase: peaks increase from `warmup_start_lr` up to `lr_max`.
2) Decay phase: peaks decrease from `lr_max` down to `lr_end`.

#### Warmup start peak
If warmup is enabled:
- `warmup_start_lr` default: `lr_end`

#### Warmup fraction
- If `warmup_steps > 0`:
  - `pw = clamp(step / warmup_steps, 0, 1)`
- else:
  - `pw = 1`

Warmup peak:
- `peak_warm = warmup_start_lr + (lr_max - warmup_start_lr) * (pw ** lr_power)`

Decay fraction after warmup:
- `remaining = max(max_steps - warmup_steps, 1)`
- If `step <= warmup_steps`:
  - `pd = 0`
- Else:
  - `pd = clamp((step - warmup_steps) / remaining, 0, 1)`

Decay peak:
- `peak_decay = lr_end + (lr_max - lr_end) * ((1 - pd) ** lr_power)`

Final peak:
- If `step < warmup_steps`:
  - `peak_lr = peak_warm`
- Else:
  - `peak_lr = peak_decay`

---

## Lower envelope

The lower envelope is the minimum LR reached by the cosine within each cycle.

### Lower envelope without warmup
Let:
- `floor_lr_max` be the initial lower-envelope value at the start of training.

Then:
- `floor_lr = lr_end + (floor_lr_max - lr_end) * ((1 - p) ** lr_power)`

This yields:
- `floor_lr(step=0) = floor_lr_max`
- `floor_lr(step=max_steps) = lr_end`

### Lower envelope with warmup
We define a two-phase envelope:
1) Warmup phase: floor increases from `floor_lr_start` up to `floor_lr_max`.
2) Decay phase: floor decreases from `floor_lr_max` down to `lr_end`.

#### Warmup start floor
If warmup is enabled:
- `floor_lr_start` default: `lr_end`

#### Maximum floor
- `floor_lr_max` must be explicitly configurable.
- For backward compatibility:
  - if `lr_floor` is not provided but `eta_min` is provided, use `eta_min` as `floor_lr_max`
  - if neither is provided, default `floor_lr_max = lr_end`

#### Warmup floor
- If `warmup_steps > 0`:
  - `pw = clamp(step / warmup_steps, 0, 1)`
- else:
  - `pw = 1`

Warmup floor:
- `floor_warm = floor_lr_start + (floor_lr_max - floor_lr_start) * (pw ** lr_power)`

Decay floor:
- `floor_decay = lr_end + (floor_lr_max - lr_end) * ((1 - pd) ** lr_power)`

Final floor:
- If `step < warmup_steps`:
  - `floor_lr = floor_warm`
- Else:
  - `floor_lr = floor_decay`

---

## Final LR computation

For each step:
1) compute `peak_lr(step)`
2) compute `floor_lr(step)`
3) enforce:
   - `floor_lr <= peak_lr`
   - if `floor_lr > peak_lr`, set `floor_lr = peak_lr`
4) compute cosine interpolation factor `alpha`
5) compute:
   - `lr = floor_lr + (peak_lr - floor_lr) * alpha`

This means the scheduler oscillates between a moving floor and a moving peak.

---

## Clamping and safety rules
- Always enforce:
  - `lr_end <= floor_lr <= peak_lr`
  - `floor_lr <= lr <= peak_lr`
- If computed `peak_lr < lr_end`, set `peak_lr = lr_end`.
- If computed `floor_lr < lr_end`, set `floor_lr = lr_end`.
- If computed `floor_lr > peak_lr`, set `floor_lr = peak_lr`.
- If user sets inconsistent values (e.g. negative LRs), raise a config validation error.

### Corner case handling
- If `warmup_steps >= max_steps`:
  - scheduler runs only warmup envelopes for the entire training, and never enters decay.
  - define `warmup_steps_effective = max_steps`
  - `pd` remains 0
  - both `peak_lr` and `floor_lr` simply warm up for the whole run.

### Resume support
- Scheduler must be fully determined by `step` (no hidden mutable cycle state),
  and must accept a `last_step` / `last_epoch` so resuming yields identical LR.

### Multi param groups
- Must support multiple optimizer param groups.
- The computed scheduler scalar is applied consistently across all groups according to the current project scheduler conventions.

---

## Backward compatibility
The previous version used a fixed minimum controlled by `eta_min`.

New behavior:
- `eta_min` should be treated as a backward-compatible alias for `lr_floor` / `floor_lr_max` where appropriate.
- If users provide only `eta_min` and no new lower-envelope parameters:
  - no warmup: lower envelope decays from `eta_min` to `lr_end`
  - with warmup: lower envelope warms from `floor_lr_start` (default `lr_end`) to `eta_min`, then decays to `lr_end`

This preserves the old mental model while extending it.

---

## Config mapping (SimpleTuner-style)

Minimum required keys:
- `--lr_scheduler: cosine_decay_peak`
- `--learning_rate` -> `lr_max`
- `--lr_end` -> `lr_end`
- `--lr_power` -> `lr_power`
- `--lr_scheduler_t0` -> `T_0`
- `--max_train_steps` -> `max_steps`

Optional keys:
- `--lr_warmup_steps` -> `warmup_steps`
- `--lr_warmup_start` -> `warmup_start_lr` (default `lr_end`)
- `--lr_floor` -> `floor_lr_max`
- `--lr_floor_start` -> `floor_lr_start` (default `lr_end`)
- `--eta_min` -> backward-compatible alias for `floor_lr_max`

If ST does not yet support `lr_floor` and `lr_floor_start`, they should be added cleanly in the same way as existing scheduler-specific arguments.

---

## Usage examples

### No warmup: both peak and floor decay to `lr_end`
```json
{
  "lr_scheduler": "cosine_decay_peak",
  "learning_rate": 0.0001,
  "lr_end": 0.00001,
  "lr_floor": 0.00004,
  "lr_power": 1.0,
  "lr_scheduler_t0": 250,
  "lr_warmup_steps": 0,
  "max_train_steps": 2000
}

Warmup over half the run: both peak and floor ramp up, then decay

{
  "lr_scheduler": "cosine_decay_peak",
  "learning_rate": 0.0001,
  "lr_end": 0.00001,
  "lr_floor": 0.00004,
  "lr_floor_start": 0.00001,
  "lr_power": 1.0,
  "lr_scheduler_t0": 250,
  "lr_warmup_steps": 1000,
  "lr_warmup_start": 0.00001,
  "max_train_steps": 2000
}

Backward-compatible example using eta_min

{
  "lr_scheduler": "cosine_decay_peak",
  "learning_rate": 0.0001,
  "lr_end": 0.00001,
  "eta_min": 0.00004,
  "lr_power": 1.0,
  "lr_scheduler_t0": 250,
  "lr_warmup_steps": 1000,
  "lr_warmup_start": 0.00001,
  "max_train_steps": 2000
}

Logging expectations

If ST supports LR logging:

* log lr each step
* optionally log peak_lr
* optionally log floor_lr
* optionally log cycle_phase and cycle_index

This is especially useful now that both envelopes evolve over time.

Acceptance criteria (must pass)

1. Bounds:

* at all steps: lr_end <= floor_lr <= lr <= peak_lr

2. No-warmup envelope behavior:

* without warmup, peak_lr is monotonically non-increasing
* without warmup, floor_lr is monotonically non-increasing
* peak_lr(step=0) == lr_max within tolerance
* floor_lr(step=0) == floor_lr_max within tolerance
* both tend to lr_end at the end of training

3. Warmup envelope behavior:

* with warmup, peak_lr is non-decreasing for step < warmup_steps and non-increasing afterwards
* with warmup, floor_lr is non-decreasing for step < warmup_steps and non-increasing afterwards
* peak_lr(step=warmup_steps) == lr_max within tolerance
* floor_lr(step=warmup_steps) == floor_lr_max within tolerance

4. Cosine behavior:

* cycle troughs hit floor_lr(step) within float tolerance
* cycle peaks hit peak_lr(step) within float tolerance

5. Corner case:

* if warmup_steps >= max_steps, scheduler does not crash and produces pure warmup envelopes for both upper and lower bounds

6. Resume:

* given identical inputs and last_step, LR sequence matches a non-interrupted run

7. Backward compatibility:

* if user provides eta_min but not lr_floor, scheduler still works correctly using eta_min as the lower-envelope maximum