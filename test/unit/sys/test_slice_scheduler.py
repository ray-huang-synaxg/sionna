#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for sionna.sys.slice_scheduler"""

import pytest
import torch

from sionna.sys import (
    SliceAwarePFScheduler,
    build_slice_prb_mask,
    get_slice_prb_ranges,
    validate_slice_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_slice_setup():
    """Return a minimal two-slice configuration used by multiple tests."""
    num_ut = 10
    num_freq_res = 52
    num_ofdm_sym = 14
    # First 5 UEs → QFI 1 (eMBB), last 5 UEs → QFI 9 (URLLC)
    qfi_per_ut = torch.tensor([1, 1, 1, 1, 1, 9, 9, 9, 9, 9], dtype=torch.int32)
    slice_prb_config = {
        1: {"num_prbs": 30, "policy": "proportional_fair"},
        9: {"num_prbs": 22, "policy": "priority"},
    }
    return num_ut, num_freq_res, num_ofdm_sym, qfi_per_ut, slice_prb_config


# ---------------------------------------------------------------------------
# get_slice_prb_ranges
# ---------------------------------------------------------------------------

class TestGetSlicePrbRanges:
    def test_two_slices_sorted(self):
        ranges = get_slice_prb_ranges({9: 22, 1: 30})
        # QFI 1 comes first because of sorted order
        assert ranges[1] == (0, 30)
        assert ranges[9] == (30, 52)

    def test_three_slices(self):
        ranges = get_slice_prb_ranges({1: 10, 2: 20, 3: 22})
        assert ranges[1] == (0, 10)
        assert ranges[2] == (10, 30)
        assert ranges[3] == (30, 52)

    def test_contiguous_no_gaps(self):
        ranges = get_slice_prb_ranges({1: 15, 9: 15})
        starts = sorted(v[0] for v in ranges.values())
        ends = sorted(v[1] for v in ranges.values())
        # No gaps: each start equals the previous end
        assert starts[1] == ends[0]

    def test_single_slice(self):
        ranges = get_slice_prb_ranges({1: 52})
        assert ranges[1] == (0, 52)


# ---------------------------------------------------------------------------
# build_slice_prb_mask
# ---------------------------------------------------------------------------

class TestBuildSlicePrbMask:
    def test_shape(self):
        qfi = torch.tensor([1, 1, 9, 9], dtype=torch.int32)
        ranges = get_slice_prb_ranges({1: 30, 9: 22})
        mask = build_slice_prb_mask(52, qfi, ranges)
        assert mask.shape == (52, 4)
        assert mask.dtype == torch.bool

    def test_correct_prb_bands(self):
        qfi = torch.tensor([1, 1, 9, 9], dtype=torch.int32)
        ranges = get_slice_prb_ranges({1: 30, 9: 22})
        mask = build_slice_prb_mask(52, qfi, ranges)
        # UEs 0-1 (QFI 1) must be True only in PRBs [0, 30)
        assert mask[:30, 0].all()
        assert mask[:30, 1].all()
        assert not mask[30:, 0].any()
        assert not mask[30:, 1].any()
        # UEs 2-3 (QFI 9) must be True only in PRBs [30, 52)
        assert mask[30:, 2].all()
        assert mask[30:, 3].all()
        assert not mask[:30, 2].any()
        assert not mask[:30, 3].any()

    def test_no_overlap_between_slices(self):
        qfi = torch.tensor([1, 9], dtype=torch.int32)
        ranges = get_slice_prb_ranges({1: 26, 9: 26})
        mask = build_slice_prb_mask(52, qfi, ranges)
        # For any PRB, at most one UE should have True (slices are disjoint)
        # mask[f, 0] XOR mask[f, 1] should always be True (exactly one is True per PRB)
        assert (mask[:, 0] ^ mask[:, 1]).all()

    def test_device_propagated(self):
        qfi = torch.tensor([1, 9], dtype=torch.int32)
        ranges = get_slice_prb_ranges({1: 26, 9: 26})
        mask = build_slice_prb_mask(52, qfi, ranges)
        assert mask.device == qfi.device


# ---------------------------------------------------------------------------
# validate_slice_config
# ---------------------------------------------------------------------------

class TestValidateSliceConfig:
    def test_valid_config_passes(self):
        qfi = torch.tensor([1, 1, 9, 9], dtype=torch.int32)
        cfg = {
            1: {"num_prbs": 30, "policy": "proportional_fair"},
            9: {"num_prbs": 22, "policy": "priority"},
        }
        validate_slice_config(qfi, cfg, 52)  # should not raise

    def test_missing_qfi_raises(self):
        qfi = torch.tensor([1, 2, 9], dtype=torch.int32)
        cfg = {
            1: {"num_prbs": 26, "policy": "proportional_fair"},
            9: {"num_prbs": 26, "policy": "priority"},
        }
        with pytest.raises(ValueError, match="QFI"):
            validate_slice_config(qfi, cfg, 52)

    def test_prb_sum_mismatch_raises(self):
        qfi = torch.tensor([1, 9], dtype=torch.int32)
        cfg = {
            1: {"num_prbs": 20, "policy": "proportional_fair"},
            9: {"num_prbs": 20, "policy": "priority"},
        }
        with pytest.raises(ValueError, match="num_prbs"):
            validate_slice_config(qfi, cfg, 52)

    def test_zero_prbs_raises(self):
        qfi = torch.tensor([1, 9], dtype=torch.int32)
        cfg = {
            1: {"num_prbs": 0, "policy": "proportional_fair"},
            9: {"num_prbs": 52, "policy": "priority"},
        }
        with pytest.raises(ValueError, match="num_prbs must be >= 1"):
            validate_slice_config(qfi, cfg, 52)

    def test_invalid_policy_raises(self):
        qfi = torch.tensor([1, 9], dtype=torch.int32)
        cfg = {
            1: {"num_prbs": 30, "policy": "best_effort"},
            9: {"num_prbs": 22, "policy": "priority"},
        }
        with pytest.raises(ValueError, match="unknown policy"):
            validate_slice_config(qfi, cfg, 52)

    def test_default_policy_is_proportional_fair(self):
        qfi = torch.tensor([1, 9], dtype=torch.int32)
        cfg = {
            1: {"num_prbs": 30},  # no "policy" key
            9: {"num_prbs": 22, "policy": "priority"},
        }
        validate_slice_config(qfi, cfg, 52)  # should not raise

    def test_all_valid_policies_accepted(self):
        for policy in ("proportional_fair", "priority", "round_robin"):
            qfi = torch.tensor([1], dtype=torch.int32)
            cfg = {1: {"num_prbs": 52, "policy": policy}}
            validate_slice_config(qfi, cfg, 52)  # should not raise


# ---------------------------------------------------------------------------
# SliceAwarePFScheduler — basic properties
# ---------------------------------------------------------------------------

class TestSliceAwarePFSchedulerBasic:
    def test_output_shape_no_batch(self, two_slice_setup, device, precision):
        num_ut, num_freq_res, num_ofdm_sym, qfi_per_ut, slice_prb_config = two_slice_setup
        qfi_per_ut = qfi_per_ut.to(device)

        scheduler = SliceAwarePFScheduler(
            qfi_per_ut=qfi_per_ut,
            slice_prb_config=slice_prb_config,
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            precision=precision,
            device=device,
        )

        rate_last_slot = torch.rand(num_ut, device=device) * 100
        rate_achievable = torch.rand(num_ofdm_sym, num_freq_res, num_ut, device=device) * 100

        result = scheduler(rate_last_slot, rate_achievable)
        assert result.shape == (num_ofdm_sym, num_freq_res, num_ut, 1)
        assert result.dtype == torch.bool

    def test_output_shape_with_batch(self, two_slice_setup, device, precision):
        num_ut, num_freq_res, num_ofdm_sym, qfi_per_ut, slice_prb_config = two_slice_setup
        batch_size = [4, 3]
        qfi_per_ut = qfi_per_ut.to(device)

        scheduler = SliceAwarePFScheduler(
            qfi_per_ut=qfi_per_ut,
            slice_prb_config=slice_prb_config,
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            batch_size=batch_size,
            precision=precision,
            device=device,
        )

        rate_last_slot = torch.rand(*batch_size, num_ut, device=device) * 100
        rate_achievable = torch.rand(
            *batch_size, num_ofdm_sym, num_freq_res, num_ut, device=device
        ) * 100

        result = scheduler(rate_last_slot, rate_achievable)
        assert result.shape == (*batch_size, num_ofdm_sym, num_freq_res, num_ut, 1)
        assert result.dtype == torch.bool

    def test_exactly_one_ut_per_prb(self, two_slice_setup, device, precision):
        """Every PRB must be allocated to exactly one UE across all OFDM symbols."""
        num_ut, num_freq_res, num_ofdm_sym, qfi_per_ut, slice_prb_config = two_slice_setup
        qfi_per_ut = qfi_per_ut.to(device)

        scheduler = SliceAwarePFScheduler(
            qfi_per_ut=qfi_per_ut,
            slice_prb_config=slice_prb_config,
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            precision=precision,
            device=device,
        )

        rate_last_slot = torch.rand(num_ut, device=device) * 100
        rate_achievable = torch.rand(num_ofdm_sym, num_freq_res, num_ut, device=device) * 100

        result = scheduler(rate_last_slot, rate_achievable)
        # sum over UE dim gives number of UEs scheduled per (sym, prb) → must be 1
        assert (result.sum(dim=-2).squeeze(-1) == 1).all()

    def test_prb_isolation_between_slices(self, two_slice_setup, device, precision):
        """UEs of slice A must never be scheduled on PRBs belonging to slice B."""
        num_ut, num_freq_res, num_ofdm_sym, qfi_per_ut, slice_prb_config = two_slice_setup
        qfi_per_ut = qfi_per_ut.to(device)

        scheduler = SliceAwarePFScheduler(
            qfi_per_ut=qfi_per_ut,
            slice_prb_config=slice_prb_config,
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            precision=precision,
            device=device,
        )

        rate_last_slot = torch.rand(num_ut, device=device) * 100
        rate_achievable = torch.rand(num_ofdm_sym, num_freq_res, num_ut, device=device) * 100

        result = scheduler(rate_last_slot, rate_achievable)
        # result: [num_ofdm_sym, num_freq_res, num_ut, 1]
        is_sched = result.squeeze(-1)  # [sym, freq, ut]

        # QFI-1 UEs (indices 0-4) must only be scheduled in PRBs [0, 30)
        qfi1_prbs_urllc = is_sched[:, 30:, :5]  # should be all-False
        assert not qfi1_prbs_urllc.any(), "QFI-1 UE scheduled on URLLC PRBs"

        # QFI-9 UEs (indices 5-9) must only be scheduled in PRBs [30, 52)
        qfi9_prbs_embb = is_sched[:, :30, 5:]   # should be all-False
        assert not qfi9_prbs_embb.any(), "QFI-9 UE scheduled on eMBB PRBs"


# ---------------------------------------------------------------------------
# SliceAwarePFScheduler — per-policy behaviour
# ---------------------------------------------------------------------------

class TestSliceAwarePFSchedulerPolicies:
    def test_priority_schedules_best_rate_ut(self, device, precision):
        """Priority policy must schedule the UE with the highest absolute rate."""
        num_ut = 4
        num_freq_res = 10
        num_ofdm_sym = 14
        # All UEs belong to one slice with 'priority' policy
        qfi_per_ut = torch.tensor([9, 9, 9, 9], dtype=torch.int32, device=device)
        slice_prb_config = {9: {"num_prbs": 10, "policy": "priority"}}

        scheduler = SliceAwarePFScheduler(
            qfi_per_ut=qfi_per_ut,
            slice_prb_config=slice_prb_config,
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            precision=precision,
            device=device,
        )

        rate_last_slot = torch.ones(num_ut, device=device)
        # UE 2 has the highest rate on every resource
        rate_achievable = torch.ones(num_ofdm_sym, num_freq_res, num_ut, device=device)
        rate_achievable[..., 2] = 1000.0

        result = scheduler(rate_last_slot, rate_achievable)
        is_sched = result.squeeze(-1)  # [sym, freq, ut]
        # UE 2 should be scheduled on every resource
        assert is_sched[..., 2].all(), "Priority: UE with highest rate not always scheduled"
        # No other UE should be scheduled
        other = is_sched[..., [0, 1, 3]]
        assert not other.any(), "Priority: other UEs should not be scheduled"

    def test_round_robin_cycles_ues(self, device, precision):
        """Round-robin must advance the UE counter each slot."""
        num_ut = 3
        num_freq_res = 6
        num_ofdm_sym = 14
        qfi_per_ut = torch.tensor([1, 1, 1], dtype=torch.int32, device=device)
        slice_prb_config = {1: {"num_prbs": 6, "policy": "round_robin"}}

        scheduler = SliceAwarePFScheduler(
            qfi_per_ut=qfi_per_ut,
            slice_prb_config=slice_prb_config,
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            precision=precision,
            device=device,
        )

        rate_last_slot = torch.ones(num_ut, device=device)
        rate_achievable = torch.rand(num_ofdm_sym, num_freq_res, num_ut, device=device)

        scheduled_ues = []
        for _ in range(3):
            result = scheduler(rate_last_slot, rate_achievable)
            is_sched = result.squeeze(-1)  # [sym, freq, ut]
            # Find which UE was scheduled (same UE on all resources in RR)
            scheduled_ut = is_sched[0, 0, :].nonzero(as_tuple=True)[0].item()
            scheduled_ues.append(scheduled_ut)

        # Over 3 slots, all 3 UEs must have been served once each
        assert sorted(scheduled_ues) == [0, 1, 2], (
            f"Round-robin did not cycle through all UEs; got {scheduled_ues}"
        )

    def test_proportional_fair_respects_history(self, device, precision):
        """PF policy should prefer UEs with lower historical rate."""
        num_ut = 2
        num_freq_res = 4
        num_ofdm_sym = 14
        qfi_per_ut = torch.tensor([1, 1], dtype=torch.int32, device=device)
        slice_prb_config = {1: {"num_prbs": 4, "policy": "proportional_fair"}}

        scheduler = SliceAwarePFScheduler(
            qfi_per_ut=qfi_per_ut,
            slice_prb_config=slice_prb_config,
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            precision=precision,
            device=device,
        )

        # Simulate UE 0 having received high historical throughput
        # by feeding a high rate_last_slot over several slots
        rate_achievable = torch.ones(num_ofdm_sym, num_freq_res, num_ut, device=device) * 10

        for _ in range(20):
            rate_last = torch.tensor([100.0, 1.0], device=device)
            scheduler(rate_last, rate_achievable)

        # Now both UEs have equal achievable rates; UE 1 (low history) should win
        rate_last = torch.tensor([0.0, 0.0], device=device)
        result = scheduler(rate_last, rate_achievable)
        is_sched = result.squeeze(-1)  # [sym, freq, ut]

        # UE 1 should be scheduled on most resources (higher PF metric due to low T_past)
        count_ut0 = is_sched[..., 0].sum().item()
        count_ut1 = is_sched[..., 1].sum().item()
        assert count_ut1 > count_ut0, (
            f"PF did not prefer UE with lower historical rate; "
            f"UE0={count_ut0}, UE1={count_ut1}"
        )


# ---------------------------------------------------------------------------
# SliceAwarePFScheduler — edge cases
# ---------------------------------------------------------------------------

class TestSliceAwarePFSchedulerEdgeCases:
    def test_single_slice_single_ut(self, device, precision):
        qfi_per_ut = torch.tensor([5], dtype=torch.int32, device=device)
        slice_prb_config = {5: {"num_prbs": 4, "policy": "proportional_fair"}}
        scheduler = SliceAwarePFScheduler(
            qfi_per_ut=qfi_per_ut,
            slice_prb_config=slice_prb_config,
            num_ut=1,
            num_freq_res=4,
            num_ofdm_sym=2,
            precision=precision,
            device=device,
        )
        rate_last = torch.ones(1, device=device)
        rate_ach = torch.rand(2, 4, 1, device=device)
        result = scheduler(rate_last, rate_ach)
        assert result.shape == (2, 4, 1, 1)
        assert result.all()  # Only one UE, must always be scheduled

    def test_three_slices(self, device, precision):
        num_ut = 9
        qfi_per_ut = torch.tensor([1, 1, 1, 2, 2, 2, 9, 9, 9], dtype=torch.int32, device=device)
        slice_prb_config = {
            1: {"num_prbs": 20, "policy": "proportional_fair"},
            2: {"num_prbs": 16, "policy": "round_robin"},
            9: {"num_prbs": 16, "policy": "priority"},
        }
        num_freq_res = 52
        scheduler = SliceAwarePFScheduler(
            qfi_per_ut=qfi_per_ut,
            slice_prb_config=slice_prb_config,
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=14,
            precision=precision,
            device=device,
        )
        rate_last = torch.rand(num_ut, device=device)
        rate_ach = torch.rand(14, num_freq_res, num_ut, device=device)
        result = scheduler(rate_last, rate_ach)
        assert result.shape == (14, num_freq_res, num_ut, 1)
        # Exactly one UE per resource
        assert (result.sum(dim=-2).squeeze(-1) == 1).all()
