# Solution Plan: Slice-Aware Scheduling

## Objective
Enable slice-aware scheduling in Sionna's system-level simulation by:
1. Associating UEs with slices via QFI (QoS Flow Indicator)
2. Configuring per-slice PRB allocation policies in the scheduler
3. Running end-to-end simulations from an external notebook to measure E2E round-trip delay and other KPIs, comparing baseline (no slicing) vs. slice-aware scheduling

---

## Architecture Overview: External Notebook vs. Sionna Source Code

This plan separates work into two clear categories:

### What lives in the **external notebook** (`Slice_Aware_Scheduling.ipynb`)
- UE-to-slice assignment (QFI tensor construction)
- Slice PRB configuration dictionaries
- PRB allocation sweep loop (varying configs across runs)
- Instantiation and invocation of `SystemLevelSimulator`
- E2E round-trip delay computation from raw simulation outputs
- KPI aggregation, comparison (baseline vs. slice-aware), and plotting

### What requires **modification inside sionna source code** (`src/sionna/sys/`)
- New file `src/sionna/sys/slice_scheduler.py`:
  - `SliceAwarePFScheduler` subclass of `PFSchedulerSUMIMO`
  - PRB mask construction utility (`build_slice_prb_mask`)
  - Slice helper functions (`validate_slice_config`, `get_slice_prb_ranges`)
- Modified `src/sionna/sys/__init__.py`:
  - Export `SliceAwarePFScheduler` and helper functions

No modifications are needed to:
- `scheduling.py` (base `PFSchedulerSUMIMO` is unchanged)
- `link_adaptation.py` (OLLA/ILLA work per-UE, unaffected)
- `power_control.py` (operates on `num_allocated_re`, naturally reflects slice mask)
- `phy_abstraction.py` (PHY abstraction is slice-agnostic)

---

## Conceptual Model

### Slices & QFI
- Each UE is assigned a **QFI** (e.g., QFI 1 = eMBB slice, QFI 9 = URLLC slice)
- A slice is a logical group of UEs sharing the same QFI
- Each slice is allocated a **PRB quota** (number of `num_freq_res` frequency resources)

### PRB Partition
```
Total PRBs = N
Slice A (QFI=1): prb_quota_A  (e.g., 30 PRBs)
Slice B (QFI=9): prb_quota_B  (e.g., 22 PRBs)
  prb_quota_A + prb_quota_B = N
```

---

## Component Design

### 1. UE-to-Slice Assignment (External Notebook)

#### 1.1 `qfi` tensor — `[num_ut_per_sector]`, `torch.int32`

**What it represents**
A 1-D integer tensor that assigns every UE within a sector to a network slice.
Index `u` holds the QFI value of UE `u`.
Example values: `1` = eMBB, `9` = URLLC, `2` = MIoT.

**Where it lives**
Constructed in the **external notebook** before simulator instantiation and passed to `SystemLevelSimulator` as a constructor argument. It does **not** change per slot — slice membership is static for the duration of one simulation run.

**Shape alignment**
The scheduler in `SystemLevelSimulator` operates per-sector with shape `[batch_size, num_bs, num_ofdm_sym, num_freq_res, num_ut_per_sector]`. The `qfi` tensor has length `num_ut_per_sector` so it indexes directly into the last dimension of scheduling tensors. The same QFI assignment is applied identically across all sectors.

**Notebook code — Construction options**
```python
# Option A — explicit per-UE list (most flexible)
qfi_per_ut = torch.tensor([1, 1, 9, 9, 1, 9, 1, 9, 1, 1], dtype=torch.int32)

# Option B — uniform split: first k UEs → QFI 1, rest → QFI 9
def split_qfi(num_ut_per_sector, qfi_a, qfi_b, frac_a=0.5):
    k = round(num_ut_per_sector * frac_a)
    return torch.cat([
        torch.full((k,),                       qfi_a, dtype=torch.int32),
        torch.full((num_ut_per_sector - k,),    qfi_b, dtype=torch.int32),
    ])

# Option C — random assignment with fixed seed (reproducible drops)
def random_qfi(num_ut_per_sector, qfi_values, seed=42):
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(qfi_values), (num_ut_per_sector,), generator=rng)
    lut = torch.tensor(qfi_values, dtype=torch.int32)
    return lut[idx]
```

**Validation rules**
- All values must be positive integers
- Every QFI in `qfi_per_ut` must appear as a key in `slice_prb_config`
- At least one UE must be assigned per QFI in `slice_prb_config`

---

### 2. Slice-Aware PRB Masking (Sionna Source Code Change)

#### 2.1 `slice_prb_config` — `dict[int, dict]`

**What it represents**
Maps each QFI to its PRB allocation policy. Each entry defines both the PRB quota (number of
frequency resources) and the allocation policy for UEs within that slice.

```python
# Example: 52 total PRBs split between two slices with different policies
slice_prb_config = {
    1: {                          # QFI 1 (eMBB)
        "num_prbs": 30,           # 30 PRBs allocated to this slice
        "policy": "proportional_fair",  # standard PF within slice
    },
    9: {                          # QFI 9 (URLLC)
        "num_prbs": 22,           # 22 PRBs allocated to this slice
        "policy": "priority",     # highest-rate UE gets first pick (low-latency)
    },
}
```

**Supported allocation policies**
| Policy | Behaviour | Use Case |
|--------|-----------|----------|
| `"proportional_fair"` | Standard PF metric `R_achievable / T_past` within the slice's PRB range | eMBB — balances throughput and fairness |
| `"priority"` | Schedule the UE with highest absolute achievable rate first (greedy) within the slice's PRB range | URLLC — minimise per-packet delay for best-channel UE |
| `"round_robin"` | Cycle through UEs in fixed order, ignoring channel quality | MIoT — guaranteed access for all devices |

**PRB band assignment (contiguous, deterministic)**
Slices are assigned contiguous non-overlapping PRB ranges.
The order of assignment follows the sorted order of QFI keys:

```
sorted QFIs → [1, 9]
QFI 1 → PRBs [0 .. 29]       (indices 0–29)
QFI 9 → PRBs [30 .. 51]      (indices 30–51)
```

**Validation rules**
- `sum(cfg["num_prbs"] for cfg in slice_prb_config.values()) == num_freq_res`
- All `num_prbs` values ≥ 1
- No duplicate QFI keys
- `policy` must be one of the supported values

---

#### 2.2 `SliceAwarePFScheduler` — New class in `src/sionna/sys/slice_scheduler.py`

**Approach**: Subclass `PFSchedulerSUMIMO` and override its `call()` method to apply per-slice PRB masking and per-slice scheduling policies.

```python
class SliceAwarePFScheduler(PFSchedulerSUMIMO):
    """Slice-aware scheduler that partitions PRBs by QFI and applies
    per-slice allocation policies.

    Parameters
    ----------
    qfi_per_ut : torch.Tensor, shape [num_ut_per_sector], dtype torch.int32
        QFI assignment for each UE within a sector.
    slice_prb_config : dict[int, dict]
        Per-slice PRB quota and scheduling policy.
        Keys are QFI values; values are dicts with:
          - "num_prbs" (int): number of PRBs for this slice
          - "policy" (str): "proportional_fair" | "priority" | "round_robin"
    (remaining args forwarded to PFSchedulerSUMIMO)
    """

    def __init__(self, qfi_per_ut, slice_prb_config, num_ut, num_freq_res,
                 num_ofdm_sym, **kwargs):
        super().__init__(num_ut, num_freq_res, num_ofdm_sym, **kwargs)
        self._qfi_per_ut = qfi_per_ut
        self._slice_prb_config = slice_prb_config
        # Pre-compute per-slice PRB ranges and UE masks
        self._slice_ranges = get_slice_prb_ranges(
            {qfi: cfg["num_prbs"] for qfi, cfg in slice_prb_config.items()})
        self._slice_ut_masks = {}   # qfi → bool tensor [num_ut]
        self._slice_policies = {}   # qfi → str
        for qfi, cfg in slice_prb_config.items():
            self._slice_ut_masks[qfi] = (qfi_per_ut == qfi)
            self._slice_policies[qfi] = cfg.get("policy", "proportional_fair")
        # Build the [num_freq_res, num_ut] PRB mask (True = allowed)
        self._prb_mask = build_slice_prb_mask(
            num_freq_res, qfi_per_ut, self._slice_ranges)

    def call(self, rate_last_slot, rate_achievable_curr_slot, **kwargs):
        """Apply per-slice masking then schedule within each slice."""
        # 1. Update average achieved rate (reuse parent logic)
        #    ... (call parent's rate update)

        # 2. Mask: zero out achievable rate for out-of-slice PRBs
        #    prb_mask shape: [num_freq_res, num_ut] → broadcast to
        #    [batch_size, num_ofdm_sym, num_freq_res, num_ut]
        rate_masked = rate_achievable_curr_slot * self._prb_mask.unsqueeze(0).unsqueeze(0)

        # 3. Within each slice, apply the configured policy:
        #    - "proportional_fair": standard PF metric on masked rates
        #    - "priority": argmax of absolute rate (no division by T_past)
        #    - "round_robin": cycle UE index per PRB

        # 4. Combine per-slice scheduling decisions into final is_scheduled
        return is_scheduled
```

#### 2.3 PRB mask utility

```python
def build_slice_prb_mask(num_freq_res, qfi_per_ut, slice_ranges):
    """Build boolean mask [num_freq_res, num_ut_per_sector].

    mask[f, u] = True iff PRB f belongs to the slice of UE u.
    """
    mask = torch.zeros(num_freq_res, len(qfi_per_ut), dtype=torch.bool)
    for qfi, (start, end) in slice_ranges.items():
        ut_in_slice = (qfi_per_ut == qfi)
        mask[start:end, :] = mask[start:end, :] | ut_in_slice.unsqueeze(0)
    return mask
```

#### 2.4 Helper functions in `src/sionna/sys/slice_scheduler.py`

```python
def validate_slice_config(qfi_per_ut, slice_prb_config, num_freq_res):
    """Raise ValueError if the slice configuration is inconsistent.

    Checks:
    - Every QFI in qfi_per_ut has a key in slice_prb_config
    - sum of num_prbs == num_freq_res
    - All num_prbs ≥ 1
    - All policies are valid
    """
    valid_policies = {"proportional_fair", "priority", "round_robin"}
    present_qfis = set(qfi_per_ut.tolist())
    configured_qfis = set(slice_prb_config.keys())
    missing = present_qfis - configured_qfis
    if missing:
        raise ValueError(f"QFIs {missing} in qfi_per_ut but no PRB config.")
    total = sum(cfg["num_prbs"] for cfg in slice_prb_config.values())
    if total != num_freq_res:
        raise ValueError(f"PRB sum ({total}) != num_freq_res ({num_freq_res}).")
    for qfi, cfg in slice_prb_config.items():
        if cfg["num_prbs"] < 1:
            raise ValueError(f"QFI {qfi}: num_prbs must be ≥ 1.")
        if cfg.get("policy", "proportional_fair") not in valid_policies:
            raise ValueError(f"QFI {qfi}: unknown policy '{cfg['policy']}'.")

def get_slice_prb_ranges(prb_quotas):
    """Convert {qfi: num_prbs} → {qfi: (start, end)} with contiguous ranges."""
    ranges = {}
    cursor = 0
    for qfi in sorted(prb_quotas):
        quota = prb_quotas[qfi]
        ranges[qfi] = (cursor, cursor + quota)
        cursor += quota
    return ranges
```

**Module location**
All classes and helpers live in `src/sionna/sys/slice_scheduler.py` and are exported via
`sionna.sys.__init__`.

---

### 3. Link Adaptation (LA) — No Source Code Changes Needed
- `InnerLoopLinkAdaptation` and `OuterLoopLinkAdaptation` work per-UE on effective SINR
- The slice PRB mask will naturally reduce the number of allocated subcarriers; LA adjusts MCS accordingly
- Track per-slice BLER / effective SINR as KPI output in the notebook

### 4. Power Control — No Source Code Changes Needed
- `open_loop_uplink_power_control` and `downlink_fair_power_control` already accept `num_allocated_re` per UE
- With the slice mask applied, the allocated RE count naturally reflects PRB quota
- Track per-slice TX power as KPI output in the notebook

---

### 5. PRB Allocation Configuration Interface (External Notebook)

The PRB allocation is configured and swept entirely from the **external notebook**. No changes
to sionna source code are needed for the sweep itself — only `SliceAwarePFScheduler` needs to
exist as a library component.

#### 5.1 Configuration data structure

```python
# Each config is a slice_prb_config dict passed to SystemLevelSimulator
sweep_configs = [
    # Config A: favour URLLC (more PRBs, priority scheduling)
    {
        1: {"num_prbs": 10, "policy": "proportional_fair"},
        9: {"num_prbs": 42, "policy": "priority"},
    },
    # Config B: equal split
    {
        1: {"num_prbs": 26, "policy": "proportional_fair"},
        9: {"num_prbs": 26, "policy": "priority"},
    },
    # Config C: favour eMBB
    {
        1: {"num_prbs": 42, "policy": "proportional_fair"},
        9: {"num_prbs": 10, "policy": "priority"},
    },
]
```

#### 5.2 Passing config to the simulator

The notebook constructs `SystemLevelSimulator` with two additional optional parameters.
When both are `None`, the simulator falls back to the original `PFSchedulerSUMIMO` (baseline
mode). This requires a small change inside `SystemLevelSimulator.__init__`:

```python
# In the notebook:
sls = SystemLevelSimulator(
    batch_size, num_rings, num_ut_per_sector,
    carrier_frequency, resource_grid, scenario, direction,
    ut_array, bs_array, bs_max_power_dbm, ut_max_power_dbm,
    coherence_time,
    # ---- new optional args ----
    qfi_per_ut=qfi_per_ut,             # torch.Tensor [num_ut_per_sector]
    slice_prb_config=slice_prb_config,  # dict[int, dict] or None
)
```

**Sionna source change in `SystemLevelSimulator.__init__`** (minimal):
```python
# Replace the scheduler construction block:
if qfi_per_ut is not None and slice_prb_config is not None:
    validate_slice_config(qfi_per_ut, slice_prb_config, resource_grid.fft_size)
    self.scheduler = SliceAwarePFScheduler(
        qfi_per_ut=qfi_per_ut,
        slice_prb_config=slice_prb_config,
        num_ut=self.num_ut_per_sector,
        num_freq_res=resource_grid.fft_size,
        num_ofdm_sym=resource_grid.num_ofdm_symbols,
        batch_size=[self.batch_size, self.num_bs],
        num_streams_per_ut=self.num_streams_per_ut,
        beta=pf_beta,
        precision=self.precision,
        device=self.device)
else:
    self.scheduler = PFSchedulerSUMIMO(...)  # existing code, unchanged
```

#### 5.3 Sweep loop in the notebook

```python
import copy

results_by_config = {}

for config_name, slice_cfg in zip(["favor_urllc", "equal", "favor_embb"], sweep_configs):
    # Build simulator with this slice config
    sls = SystemLevelSimulator(
        batch_size, num_rings, num_ut_per_sector,
        carrier_frequency, resource_grid, scenario, direction,
        ut_array, bs_array, bs_max_power_dbm, ut_max_power_dbm,
        coherence_time,
        qfi_per_ut=qfi_per_ut,
        slice_prb_config=slice_cfg,
    )

    # Run simulation
    hist = sls(num_slots, alpha_ul, p0_dbm_ul, bler_target, olla_delta_up)
    hist = clean_hist(hist)

    # Store per-config results
    results_by_config[config_name] = hist
```

#### 5.4 Baseline run (no slice-aware scheduling)

```python
# Baseline: pass qfi_per_ut=None, slice_prb_config=None
sls_baseline = SystemLevelSimulator(
    batch_size, num_rings, num_ut_per_sector,
    carrier_frequency, resource_grid, scenario, direction,
    ut_array, bs_array, bs_max_power_dbm, ut_max_power_dbm,
    coherence_time,
    qfi_per_ut=None,
    slice_prb_config=None,
)
hist_baseline = sls_baseline(num_slots, alpha_ul, p0_dbm_ul, bler_target, olla_delta_up)
hist_baseline = clean_hist(hist_baseline)
```

---

## E2E Round-Trip Delay Metrics (External Notebook)

### 6.1 What Sionna can measure

Sionna's system-level simulator does **not** model a full protocol stack (no RLC, no MAC HARQ
process timing, no core-network delay). However, the simulation outputs allow **estimating
the radio-access-network (RAN) contribution to E2E delay** on a per-UE, per-slot basis.

The following raw metrics are available from the `hist` dictionary returned by
`SystemLevelSimulator.call()`:

| `hist` key | Shape (per slot) | Description |
|---|---|---|
| `harq` | `[num_bs, num_ut_per_sector]` | HARQ feedback: 1=ACK, 0=NACK, NaN=not scheduled |
| `num_decoded_bits` | `[num_bs, num_ut_per_sector]` | Successfully decoded bits this slot |
| `num_allocated_re` | `[num_bs, num_ut_per_sector]` | Number of resource elements allocated |
| `mcs_index` | `[num_bs, num_ut_per_sector]` | MCS index used |
| `sinr_eff` | `[num_bs, num_ut_per_sector]` | Effective SINR (dB) |
| `pf_metric` | `[num_bs, num_ut_per_sector]` | PF scheduling metric |

### 6.2 Derived delay metrics (computed in notebook)

#### 6.2.1 Scheduling delay (waiting time)

The number of consecutive slots a UE waits before being scheduled:

```python
def compute_scheduling_delay(hist, num_slots, num_bs, num_ut_per_sector, slot_duration_ms):
    """Compute per-UE scheduling delay: slots waited between consecutive allocations.

    Returns
    -------
    sched_delay : np.ndarray, shape [num_bs, num_ut_per_sector]
        Mean scheduling delay in milliseconds per UE.
    """
    # harq: [num_slots, num_bs, num_ut_per_sector]
    is_scheduled = ~np.isnan(hist['harq'])  # True when UE was scheduled

    delays = np.full((num_bs, num_ut_per_sector), np.nan)
    for bs in range(num_bs):
        for ue in range(num_ut_per_sector):
            sched_slots = np.where(is_scheduled[:, bs, ue])[0]
            if len(sched_slots) > 1:
                inter_sched_gaps = np.diff(sched_slots)
                delays[bs, ue] = np.mean(inter_sched_gaps) * slot_duration_ms
    return delays
```

#### 6.2.2 Transmission delay (air interface)

Time to successfully transmit a transport block, accounting for HARQ retransmissions:

```python
def compute_transmission_delay(hist, num_slots, num_bs, num_ut_per_sector, slot_duration_ms):
    """Estimate transmission delay per UE including HARQ retransmissions.

    A NACK (harq=0) means the TB must be retransmitted, adding at least one
    slot of delay. Consecutive NACKs before an ACK are counted.

    Returns
    -------
    tx_delay : np.ndarray, shape [num_bs, num_ut_per_sector]
        Mean transmission delay in milliseconds per UE.
    """
    harq = hist['harq']  # [num_slots, num_bs, num_ut_per_sector]
    delays = np.full((num_bs, num_ut_per_sector), np.nan)

    for bs in range(num_bs):
        for ue in range(num_ut_per_sector):
            harq_seq = harq[:, bs, ue]
            sched_mask = ~np.isnan(harq_seq)
            sched_slots = np.where(sched_mask)[0]
            if len(sched_slots) == 0:
                continue

            # Count consecutive NACKs before each ACK
            tb_delays = []
            nack_count = 0
            for s in sched_slots:
                if harq_seq[s] == 0:      # NACK
                    nack_count += 1
                elif harq_seq[s] == 1:    # ACK
                    # delay = 1 slot (initial tx) + nack_count retx slots
                    tb_delays.append((1 + nack_count) * slot_duration_ms)
                    nack_count = 0

            if tb_delays:
                delays[bs, ue] = np.mean(tb_delays)
    return delays
```

#### 6.2.3 E2E round-trip delay estimate

Combines scheduling delay + transmission delay as the RAN-level round-trip time:

```python
def compute_e2e_delay(hist, num_slots, num_bs, num_ut_per_sector, slot_duration_ms):
    """Estimate E2E round-trip delay per UE.

    E2E delay = scheduling_delay + transmission_delay (downlink)
              + scheduling_delay + transmission_delay (uplink, if applicable)

    For single-direction simulation:
        E2E_one_way = scheduling_delay + transmission_delay
        E2E_round_trip ≈ 2 × E2E_one_way  (symmetric assumption)

    Returns
    -------
    e2e_delay : np.ndarray, shape [num_bs, num_ut_per_sector]
        Estimated mean E2E round-trip delay in milliseconds per UE.
    """
    sched_delay = compute_scheduling_delay(
        hist, num_slots, num_bs, num_ut_per_sector, slot_duration_ms)
    tx_delay = compute_transmission_delay(
        hist, num_slots, num_bs, num_ut_per_sector, slot_duration_ms)

    one_way = np.nansum([sched_delay, tx_delay], axis=0)
    round_trip = 2.0 * one_way  # symmetric assumption for DL+UL
    return round_trip, sched_delay, tx_delay
```

### 6.3 Scenario comparison: Baseline vs. Slice-Aware

The notebook runs both scenarios and compares delay metrics per UE:

#### Scenario 1 — Baseline (no slice-aware scheduling)
```python
# Baseline run
sls_baseline = SystemLevelSimulator(
    ..., qfi_per_ut=None, slice_prb_config=None)
hist_baseline = clean_hist(sls_baseline(num_slots, ...))

e2e_baseline, sched_baseline, tx_baseline = compute_e2e_delay(
    hist_baseline, num_slots, num_bs, num_ut_per_sector, slot_duration_ms)
```

#### Scenario 2 — With slice-aware scheduling
```python
# Slice-aware run
sls_sliced = SystemLevelSimulator(
    ..., qfi_per_ut=qfi_per_ut, slice_prb_config=slice_prb_config)
hist_sliced = clean_hist(sls_sliced(num_slots, ...))

e2e_sliced, sched_sliced, tx_sliced = compute_e2e_delay(
    hist_sliced, num_slots, num_bs, num_ut_per_sector, slot_duration_ms)
```

#### Per-slice aggregation and comparison
```python
# Group UEs by QFI for per-slice statistics
qfi_np = qfi_per_ut.cpu().numpy()

for qfi_val in np.unique(qfi_np):
    ue_mask = (qfi_np == qfi_val)
    slice_name = {1: "eMBB", 9: "URLLC"}.get(qfi_val, f"QFI={qfi_val}")

    print(f"\n=== {slice_name} (QFI={qfi_val}) ===")
    print(f"  Baseline E2E RTT:        {np.nanmean(e2e_baseline[:, ue_mask]):.2f} ms")
    print(f"  Slice-aware E2E RTT:     {np.nanmean(e2e_sliced[:, ue_mask]):.2f} ms")
    print(f"  Baseline sched delay:    {np.nanmean(sched_baseline[:, ue_mask]):.2f} ms")
    print(f"  Slice-aware sched delay: {np.nanmean(sched_sliced[:, ue_mask]):.2f} ms")
    print(f"  Baseline tx delay:       {np.nanmean(tx_baseline[:, ue_mask]):.2f} ms")
    print(f"  Slice-aware tx delay:    {np.nanmean(tx_sliced[:, ue_mask]):.2f} ms")
```

### 6.4 Full KPI table (per slice, per scenario)

| KPI | Source | Baseline | Slice-Aware | How to fetch from `hist` |
|-----|--------|----------|-------------|--------------------------|
| **E2E round-trip delay** | Derived | ✅ | ✅ | `compute_e2e_delay(hist, ...)` |
| **Scheduling delay** | Derived | ✅ | ✅ | `compute_scheduling_delay(hist, ...)` |
| **Transmission delay** | Derived (HARQ) | ✅ | ✅ | `compute_transmission_delay(hist, ...)` |
| **Throughput** (bits/slot) | `hist['num_decoded_bits']` | ✅ | ✅ | `np.nanmean(hist['num_decoded_bits'][:, :, ue_mask], axis=0)` |
| **TBLER** | `hist['harq']` | ✅ | ✅ | `1 - np.nanmean(hist['harq'][:, :, ue_mask], axis=0)` |
| **MCS index** | `hist['mcs_index']` | ✅ | ✅ | `np.nanmean(hist['mcs_index'][:, :, ue_mask], axis=0)` |
| **Effective SINR** | `hist['sinr_eff']` | ✅ | ✅ | `np.nanmean(hist['sinr_eff'][:, :, ue_mask], axis=0)` |
| **OLLA offset** | `hist['olla_offset']` | ✅ | ✅ | `np.nanmean(hist['olla_offset'][:, :, ue_mask], axis=0)` |
| **TX power** | `hist['tx_power']` | ✅ | ✅ | `np.nanmean(hist['tx_power'][:, :, ue_mask], axis=0)` |
| **PRB utilization** | `hist['num_allocated_re']` | ✅ | ✅ | `np.nansum(hist['num_allocated_re'][:, :, ue_mask]) / (num_prbs × num_ofdm_sym × num_slots)` |
| **PF metric** | `hist['pf_metric']` | ✅ | ✅ | `np.nanmean(hist['pf_metric'][:, :, ue_mask], axis=0)` |

All KPIs are available per-UE from `hist` and can be grouped by QFI in the notebook to
produce per-slice statistics.

### 6.5 Slot duration reference

```python
# slot_duration_ms depends on the resource grid configuration
# For subcarrier_spacing = 15 kHz, 1 slot = 14 OFDM symbols ≈ 1 ms
# For subcarrier_spacing = 30 kHz, 1 slot ≈ 0.5 ms
slot_duration_ms = resource_grid.ofdm_symbol_duration * resource_grid.num_ofdm_symbols * 1e3
```

---

## Implementation Steps (ordered)

1. **Sionna source — `slice_scheduler.py`**: Create `src/sionna/sys/slice_scheduler.py` with:
   - `SliceAwarePFScheduler` class
   - `build_slice_prb_mask()` utility
   - `validate_slice_config()` helper
   - `get_slice_prb_ranges()` helper

2. **Sionna source — `__init__.py`**: Export new symbols from `sionna.sys`

3. **Sionna source — `SystemLevelSimulator` support** (in E2E notebook class or factored out):
   Accept optional `qfi_per_ut` and `slice_prb_config` parameters; conditionally use `SliceAwarePFScheduler`

4. **Unit tests**: `test/unit/sys/test_slice_scheduler.py` — PRB mask correctness, policy behaviour, validation errors

5. **External notebook** `tutorials/sys/Slice_Aware_Scheduling.ipynb`:
   - Scenario setup (topology, antennas, resource grid)
   - QFI assignment and slice_prb_config construction
   - Baseline run (no slicing)
   - Slice-aware run(s) with configurable PRB splits and policies
   - E2E round-trip delay computation per UE
   - Per-slice KPI comparison plots (CDF of delay, throughput, BLER)

6. **Validation**: Confirm baseline run produces identical results to original `End-to-End_Example.ipynb`

---

## Deliverables
- `src/sionna/sys/slice_scheduler.py` — `SliceAwarePFScheduler` + PRB mask utility + helpers
- `test/unit/sys/test_slice_scheduler.py` — unit tests for mask logic, policy behaviour, validation
- `tutorials/sys/Slice_Aware_Scheduling.ipynb` — notebook: baseline vs. slice-aware → E2E delay comparison → KPI plots
