#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Slice-aware scheduler for Sionna SYS"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from sionna.phy.utils import insert_dims
from sionna.sys.scheduling import PFSchedulerSUMIMO

__all__ = [
    "SliceAwarePFScheduler",
    "build_slice_prb_mask",
    "validate_slice_config",
    "get_slice_prb_ranges",
]

VALID_POLICIES = {"proportional_fair", "priority", "round_robin"}


def validate_slice_config(
    qfi_per_ut: torch.Tensor,
    slice_prb_config: Dict[int, Dict],
    num_freq_res: int,
) -> None:
    """Validate that the slice configuration is consistent.

    Parameters
    ----------
    qfi_per_ut : torch.Tensor, shape [num_ut_per_sector], dtype torch.int32
        QFI assignment for each UE.
    slice_prb_config : dict[int, dict]
        Per-slice PRB quota and scheduling policy.
    num_freq_res : int
        Total number of frequency resources.

    Raises
    ------
    ValueError
        If any consistency check fails.
    """
    present_qfis = set(int(q) for q in qfi_per_ut.tolist())
    configured_qfis = set(slice_prb_config.keys())
    missing = present_qfis - configured_qfis
    if missing:
        raise ValueError(
            f"QFIs {missing} appear in qfi_per_ut but have no entry in slice_prb_config."
        )

    total_prbs = sum(cfg["num_prbs"] for cfg in slice_prb_config.values())
    if total_prbs != num_freq_res:
        raise ValueError(
            f"Sum of num_prbs across slices ({total_prbs}) does not equal "
            f"num_freq_res ({num_freq_res})."
        )

    for qfi, cfg in slice_prb_config.items():
        if cfg["num_prbs"] < 1:
            raise ValueError(
                f"QFI {qfi}: num_prbs must be >= 1, got {cfg['num_prbs']}."
            )
        policy = cfg.get("policy", "proportional_fair")
        if policy not in VALID_POLICIES:
            raise ValueError(
                f"QFI {qfi}: unknown policy '{policy}'. "
                f"Valid options: {VALID_POLICIES}."
            )


def get_slice_prb_ranges(
    prb_quotas: Dict[int, int],
) -> Dict[int, Tuple[int, int]]:
    """Convert per-slice PRB quotas to contiguous, non-overlapping ranges.

    Slices are ordered by ascending QFI key.

    Parameters
    ----------
    prb_quotas : dict[int, int]
        Mapping from QFI to PRB count.

    Returns
    -------
    dict[int, tuple[int, int]]
        Mapping from QFI to (start_prb, end_prb) where the range is
        ``[start_prb, end_prb)`` (half-open).
    """
    ranges: Dict[int, Tuple[int, int]] = {}
    cursor = 0
    for qfi in sorted(prb_quotas):
        quota = prb_quotas[qfi]
        ranges[qfi] = (cursor, cursor + quota)
        cursor += quota
    return ranges


def build_slice_prb_mask(
    num_freq_res: int,
    qfi_per_ut: torch.Tensor,
    slice_ranges: Dict[int, Tuple[int, int]],
) -> torch.Tensor:
    """Build a boolean mask allowing each UE only on its slice's PRBs.

    Parameters
    ----------
    num_freq_res : int
        Total number of frequency resources.
    qfi_per_ut : torch.Tensor, shape [num_ut_per_sector], dtype torch.int32
        QFI assignment for each UE.
    slice_ranges : dict[int, tuple[int, int]]
        Mapping from QFI to (start_prb, end_prb), as returned by
        :func:`get_slice_prb_ranges`.

    Returns
    -------
    torch.Tensor, shape [num_freq_res, num_ut_per_sector], dtype torch.bool
        ``mask[f, u]`` is ``True`` iff frequency resource ``f`` belongs to
        the slice of UE ``u``.
    """
    num_ut = len(qfi_per_ut)
    mask = torch.zeros(num_freq_res, num_ut, dtype=torch.bool, device=qfi_per_ut.device)
    for qfi, (start, end) in slice_ranges.items():
        ut_in_slice = (qfi_per_ut == qfi)  # [num_ut]
        # Broadcast: set mask True for all PRBs in [start, end) for UEs in this slice
        mask[start:end, :] = mask[start:end, :] | ut_in_slice.unsqueeze(0)
    return mask


class SliceAwarePFScheduler(PFSchedulerSUMIMO):
    r"""Slice-aware scheduler that partitions PRBs by QFI and applies
    per-slice allocation policies.

    Subclasses :class:`~sionna.sys.PFSchedulerSUMIMO`. Each network slice
    (identified by its QFI value) receives a contiguous, non-overlapping
    range of frequency resources. Within each slice, one of three
    allocation policies can be applied:

    * ``"proportional_fair"`` – standard PF metric :math:`\tilde{R}/T_\text{past}`.
    * ``"priority"`` – schedule the UE with the highest absolute achievable
      rate (greedy, no division by :math:`T_\text{past}`).
    * ``"round_robin"`` – cycle through UEs in fixed (index) order.

    Parameters
    ----------
    qfi_per_ut : torch.Tensor, shape [num_ut_per_sector], dtype torch.int32
        QFI assignment for each UE within a sector.
    slice_prb_config : dict[int, dict]
        Per-slice PRB quota and scheduling policy.  Keys are QFI values;
        each value is a dict with:

        * ``"num_prbs"`` *(int)* – number of PRBs for this slice.
        * ``"policy"`` *(str, optional)* – ``"proportional_fair"`` (default),
          ``"priority"``, or ``"round_robin"``.

    num_ut : int
        Number of user terminals (must equal ``len(qfi_per_ut)``).
    num_freq_res : int
        Total number of frequency resources.
    num_ofdm_sym : int
        Number of OFDM symbols per slot.
    batch_size : list[int] or int or None
        Batch size / shape (e.g. ``[batch, num_bs]``).
    num_streams_per_ut : int, optional
        Number of MIMO streams per UE. Defaults to 1.
    beta : float, optional
        PF discount factor. Defaults to 0.98.
    precision : optional
        Numerical precision. Defaults to global config.
    device : str or None, optional
        Computation device.

    :input rate_last_slot: [batch_size, num_ut], ``torch.float``
        Rate achieved by each UE in the previous slot.
    :input rate_achievable_curr_slot: [batch_size, num_ofdm_sym, num_freq_res, num_ut], ``torch.float``
        Achievable rate for each UE on each time-frequency resource.

    :output is_scheduled: [batch_size, num_ofdm_sym, num_freq_res, num_ut, num_streams_per_ut], ``torch.bool``
        Scheduling decisions with per-slice PRB masking applied.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.sys import SliceAwarePFScheduler

        num_ut = 10
        num_freq_res = 52
        num_ofdm_sym = 14

        qfi_per_ut = torch.tensor([1,1,1,1,1,9,9,9,9,9], dtype=torch.int32)
        slice_prb_config = {
            1: {"num_prbs": 30, "policy": "proportional_fair"},
            9: {"num_prbs": 22, "policy": "priority"},
        }

        scheduler = SliceAwarePFScheduler(
            qfi_per_ut=qfi_per_ut,
            slice_prb_config=slice_prb_config,
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
        )

        rate_last_slot = torch.rand(num_ut) * 100
        rate_achievable = torch.rand(num_ofdm_sym, num_freq_res, num_ut) * 100
        is_scheduled = scheduler(rate_last_slot, rate_achievable)
        print(is_scheduled.shape)
        # torch.Size([14, 52, 10, 1])
    """

    def __init__(
        self,
        qfi_per_ut: torch.Tensor,
        slice_prb_config: Dict[int, Dict],
        num_ut: int,
        num_freq_res: int,
        num_ofdm_sym: int,
        batch_size: Optional[Union[List[int], int]] = None,
        num_streams_per_ut: int = 1,
        beta: float = 0.98,
        precision=None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            num_ut=num_ut,
            num_freq_res=num_freq_res,
            num_ofdm_sym=num_ofdm_sym,
            batch_size=batch_size,
            num_streams_per_ut=num_streams_per_ut,
            beta=beta,
            precision=precision,
            device=device,
        )

        self._qfi_per_ut = qfi_per_ut
        self._slice_prb_config = slice_prb_config

        # Pre-compute contiguous PRB ranges {qfi: (start, end)}
        self._slice_ranges = get_slice_prb_ranges(
            {qfi: cfg["num_prbs"] for qfi, cfg in slice_prb_config.items()}
        )

        # Pre-compute per-slice UE membership masks and policies
        self._slice_ut_masks: Dict[int, torch.Tensor] = {}
        self._slice_policies: Dict[int, str] = {}
        for qfi, cfg in slice_prb_config.items():
            self._slice_ut_masks[qfi] = qfi_per_ut == qfi  # [num_ut], bool
            self._slice_policies[qfi] = cfg.get("policy", "proportional_fair")

        # Build [num_freq_res, num_ut] boolean PRB mask
        # mask[f, u] = True  iff PRB f belongs to UE u's slice
        self._prb_mask = build_slice_prb_mask(
            num_freq_res, qfi_per_ut, self._slice_ranges
        )

        # Round-robin counter per slice (index of the next UE to serve)
        self._rr_counters: Dict[int, int] = {qfi: 0 for qfi in slice_prb_config}

    @property
    def slice_ranges(self) -> Dict[int, Tuple[int, int]]:
        """dict[int, tuple[int, int]] (read-only): PRB range per QFI."""
        return self._slice_ranges

    @property
    def prb_mask(self) -> torch.Tensor:
        """[num_freq_res, num_ut] (read-only): PRB allocation mask."""
        return self._prb_mask

    def call(
        self,
        rate_last_slot: torch.Tensor,
        rate_achievable_curr_slot: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute slice-aware scheduling decisions.

        Parameters
        ----------
        rate_last_slot :
            [batch_size, num_ut] rates achieved in the previous slot.
        rate_achievable_curr_slot :
            [batch_size, num_ofdm_sym, num_freq_res, num_ut] achievable rates.

        Returns
        -------
        torch.Tensor
            [batch_size, num_ofdm_sym, num_freq_res, num_ut, num_streams_per_ut],
            ``torch.bool``.
        """
        # --- cast inputs ---
        rate_last_slot = rate_last_slot.to(self.dtype)
        rate_achievable_curr_slot = rate_achievable_curr_slot.to(self.dtype)

        # --- update average achieved rate (identical to base class) ---
        beta = self._beta_value
        rate_achieved_past_new = (
            beta * self._rate_achieved_past + (1 - beta) * rate_last_slot
        )
        self._rate_achieved_past.copy_(rate_achieved_past_new)

        # [batch_size, 1, 1, num_ut]
        rate_achieved_past = insert_dims(rate_achieved_past_new, 2, axis=-2)

        # --- apply PRB mask: zero achievable rate for out-of-slice PRBs ---
        # _prb_mask: [num_freq_res, num_ut] → broadcast to full grid shape
        prb_mask = self._prb_mask.to(rate_achievable_curr_slot.device)

        # Determine leading batch dims by counting extra dims beyond [sym, freq, ut]
        ndim = rate_achievable_curr_slot.dim()
        # shape: [num_freq_res, num_ut] needs ndim-3 leading dims and 0 trailing
        for _ in range(ndim - 3):
            prb_mask = prb_mask.unsqueeze(0)
        # prb_mask: [1, ..., num_freq_res, num_ut] broadcastable to full shape
        # Insert OFDM-symbol dim (second-to-last of the original 4-D block)
        prb_mask = prb_mask.unsqueeze(-3)  # [..., 1, num_freq_res, num_ut]

        rate_masked = rate_achievable_curr_slot * prb_mask

        # --- PF metric for the full grid ---
        # [batch_size, num_ofdm_sym, num_freq_res, num_ut]
        pf_metric = rate_masked / rate_achieved_past

        # Store for external inspection
        self._pf_metric.copy_(pf_metric)

        # --- schedule within each slice ---
        # Accumulate scheduling decisions: start with zeros (no one scheduled)
        is_scheduled_int = torch.zeros(
            *rate_achievable_curr_slot.shape,
            dtype=torch.long,
            device=rate_achievable_curr_slot.device,
        )

        for qfi, (start, end) in self._slice_ranges.items():
            ut_mask = self._slice_ut_masks[qfi].to(rate_achievable_curr_slot.device)
            policy = self._slice_policies[qfi]

            # Indices of UEs belonging to this slice
            ut_indices = ut_mask.nonzero(as_tuple=True)[0]  # [num_ut_in_slice]
            num_ut_in_slice = len(ut_indices)
            if num_ut_in_slice == 0:
                continue

            if policy == "proportional_fair":
                # Standard PF: argmax of R̃/T_past restricted to slice PRBs and UEs
                # [batch..., num_ofdm_sym, slice_prbs, num_ut_in_slice]
                slice_metric = pf_metric[..., start:end, :][..., ut_mask]
                # [batch..., num_ofdm_sym, slice_prbs]
                local_winner = torch.argmax(slice_metric, dim=-1)
                # Map local index back to global UE index
                global_winner = ut_indices[local_winner]

            elif policy == "priority":
                # Greedy: schedule UE with highest absolute achievable rate
                slice_rate = rate_achievable_curr_slot[..., start:end, :][..., ut_mask]
                local_winner = torch.argmax(slice_rate, dim=-1)
                global_winner = ut_indices[local_winner]

            elif policy == "round_robin":
                # Cycle through UEs in fixed order, ignoring channel quality
                rr_idx = self._rr_counters[qfi] % num_ut_in_slice
                # Same UE assigned to every PRB/symbol in this slot
                global_winner = ut_indices[rr_idx].expand(
                    *rate_achievable_curr_slot.shape[:-1][:-1],  # batch + sym
                    end - start,  # num prbs in slice
                )
                self._rr_counters[qfi] = (rr_idx + 1) % num_ut_in_slice

            else:
                raise ValueError(f"Unknown policy '{policy}' for QFI {qfi}.")

            # Write scheduling decisions for this slice's PRBs
            # global_winner: [..., num_ofdm_sym, slice_prbs] with values in [0, num_ut)
            slice_scheduled = torch.nn.functional.one_hot(
                global_winner, num_classes=self._num_ut
            )  # [..., num_ofdm_sym, slice_prbs, num_ut]
            is_scheduled_int[..., start:end, :] = slice_scheduled

        # Expand to include streams dimension
        # [batch..., num_ofdm_sym, num_freq_res, num_ut, 1]
        is_scheduled = is_scheduled_int.unsqueeze(-1)
        # [batch..., num_ofdm_sym, num_freq_res, num_ut, num_streams]
        is_scheduled = is_scheduled.expand(
            *is_scheduled.shape[:-1], self._num_streams_per_ut
        )
        return is_scheduled.to(torch.bool)
