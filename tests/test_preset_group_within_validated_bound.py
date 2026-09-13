"""
If you pick Variant A (fixed b_batch_group per preset), this test is
the tripwire for preset/formula drift: fails loudly instead of OOMing
quietly on a longer sequence length than train_shape was tuned for.
"""
import pytest
from Atomic_ops.configs import KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE
from Atomic_ops.gdn2_fwd_batched import _max_validated_group


@pytest.mark.parametrize("preset,name", [
    (KAGGLE_SMALL, "KAGGLE_SMALL"),
    (KAGGLE_MEDIUM, "KAGGLE_MEDIUM"),
    (KAGGLE_LARGE, "KAGGLE_LARGE"),
])
def test_preset_group_does_not_exceed_validated_bound(preset, name):
    assert preset.b_batch_group is not None, (
        f"{name}: b_batch_group is None -- falls back to group=n_chunks, "
        f"which OOM'd at n_chunks=64/group=128 in MB8_status_report.md. "
        f"Variant A requires every preset to set this explicitly."
    )
    bound = _max_validated_group(preset.bt)
    assert preset.b_batch_group <= bound, (
        f"{name}: b_batch_group={preset.b_batch_group} exceeds "
        f"_max_validated_group(bt={preset.bt})={bound}. Either the "
        f"preset drifted from the formula, or the formula's anchor "
        f"point needs a fresh TPU measurement at this bt."
    )
