"""Tests for entitlement_counter."""

import math

import pytest
import torch

from entitlement_counter import (
    EntitlementCounter,
    EntitlementReport,
    SpeculativeDraft,
    expected_accepted_tokens,
    kv_bytes_from_args,
    profile_module,
    speculative_report,
    total_weight_bytes,
    weight_bytes_by_module,
)


class TinyMLP(torch.nn.Module):
    """Two Linear layers; FLOPs are analytically known from the shapes."""

    def __init__(self, d_in: int = 16, d_hidden: int = 32, d_out: int = 8) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = torch.nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TinyAttn(torch.nn.Module):
    """Single fused scaled-dot-product attention call over (B, H, S, D)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, q, k, v):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        b, h, s, d = out.shape
        return self.proj(out.reshape(b, h, s, d))


# ---------------------------------------------------------------------------
# Weight-byte accounting (no dispatch needed)
# ---------------------------------------------------------------------------


def test_weight_bytes_by_module_attributes_each_param_once():
    model = TinyMLP(d_in=16, d_hidden=32, d_out=8)
    by_mod = weight_bytes_by_module(model)
    # fc1: 16*32 params, fc2: 32*8 params, both float32 (4 bytes), no bias.
    assert by_mod["fc1"] == 16 * 32 * 4
    assert by_mod["fc2"] == 32 * 8 * 4
    # root module owns no params directly
    assert "" not in by_mod
    assert sum(by_mod.values()) == total_weight_bytes(model)


def test_total_weight_bytes_dedupes_tied_params():
    model = TinyMLP(d_in=8, d_hidden=8, d_out=8)
    model.fc2.weight = model.fc1.weight  # tie them
    # Deduped total counts the shared weight once.
    assert total_weight_bytes(model) == 8 * 8 * 4
    # Per-module walk still attributes the (shared) param to both owners.
    by_mod = weight_bytes_by_module(model)
    assert by_mod["fc1"] == 8 * 8 * 4
    assert by_mod["fc2"] == 8 * 8 * 4


def test_weight_bytes_work_on_meta_device():
    with torch.device("meta"):
        model = TinyMLP(d_in=64, d_hidden=128, d_out=64)
    assert total_weight_bytes(model) == (64 * 128 + 128 * 64) * 4


# ---------------------------------------------------------------------------
# KV-byte helper
# ---------------------------------------------------------------------------


def test_kv_bytes_from_args_sums_key_and_value():
    q = torch.empty(1, 2, 4, 8)
    k = torch.empty(1, 2, 4, 8)
    v = torch.empty(1, 2, 4, 8)
    expected = k.numel() * k.element_size() + v.numel() * v.element_size()
    assert kv_bytes_from_args((q, k, v)) == expected


def test_kv_bytes_from_args_ignores_non_tensors_and_missing():
    assert kv_bytes_from_args(()) == 0
    assert kv_bytes_from_args((torch.empty(2, 2),)) == 0  # only query present
    assert kv_bytes_from_args((torch.empty(2, 2), None, "x")) == 0


# ---------------------------------------------------------------------------
# End-to-end FLOP counting
# ---------------------------------------------------------------------------


def test_profile_mlp_flops_match_analytic():
    model = TinyMLP(d_in=16, d_hidden=32, d_out=8)
    x = torch.randn(4, 16)  # batch of 4
    report = profile_module(model, args=(x,))
    # mm FLOPs = 2 * m * k * n per layer; fc1: 4x16x32, fc2: 4x32x8.
    expected = 2 * 4 * 16 * 32 + 2 * 4 * 32 * 8
    assert report.total_flops == expected
    assert report.total_kv_bytes == 0  # no attention op
    assert report.total_weight_bytes == (16 * 32 + 32 * 8) * 4
    assert isinstance(report, EntitlementReport)


def test_profile_mlp_on_meta_device():
    with torch.device("meta"):
        model = TinyMLP(d_in=128, d_hidden=256, d_out=128)
        x = torch.randn(2, 128)
    report = profile_module(model, args=(x,))
    assert report.total_flops == 2 * 2 * 128 * 256 + 2 * 2 * 256 * 128
    assert report.total_weight_bytes == (128 * 256 + 256 * 128) * 4


def test_profile_attention_captures_kv_bytes():
    model = TinyAttn()
    b, h, s, d = 1, 2, 4, 8
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    report = profile_module(model, args=(q, k, v))
    expected_kv = 2 * (b * h * s * d) * 4  # key + value, float32
    assert report.total_kv_bytes == expected_kv
    assert report.total_flops > 0  # sdpa + proj mm both counted
    assert report.arithmetic_intensity > 0


def test_arithmetic_intensity_zero_when_no_traffic():
    report = EntitlementReport(total_flops=100, total_weight_bytes=0, total_kv_bytes=0)
    assert report.arithmetic_intensity == 0.0
    assert report.total_hbm_bytes == 0


def test_report_requires_model():
    counter = EntitlementCounter(model=None)
    with counter:
        pass
    try:
        counter.report()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_summary_is_readable():
    report = EntitlementReport(total_flops=2048, total_weight_bytes=1024, total_kv_bytes=512)
    text = report.summary()
    assert "FLOPs=2,048" in text
    assert "AI=" in text


# ---------------------------------------------------------------------------
# Speculative decoding (optional extension)
# ---------------------------------------------------------------------------


def test_expected_accepted_tokens_matches_truncated_geometric():
    # alpha=0.5, gamma=4 -> (1-0.5**5)/(1-0.5) = (1-1/32)/0.5 = 1.9375
    assert expected_accepted_tokens(4, 0.5) == pytest.approx(1.9375)
    # alpha=0 -> only the bonus token ever accepts -> N=1
    assert expected_accepted_tokens(3, 0.0) == pytest.approx(1.0)
    # alpha=1 -> every proposed token accepts plus the bonus -> gamma+1
    assert expected_accepted_tokens(4, 1.0) == pytest.approx(5.0)


def test_expected_accepted_tokens_bounded_and_saturates():
    # capped at gamma+1, and approaches 1/(1-alpha) as gamma grows
    assert expected_accepted_tokens(2, 0.9) <= 3.0
    assert expected_accepted_tokens(1000, 0.8) == pytest.approx(1.0 / (1.0 - 0.8), rel=1e-6)


def test_expected_accepted_tokens_validates_inputs():
    with pytest.raises(ValueError):
        expected_accepted_tokens(0, 0.5)  # gamma must be >= 1
    with pytest.raises(ValueError):
        expected_accepted_tokens(2, 1.5)  # alpha out of [0,1]


def test_speculative_report_amortizes_per_accepted_token():
    # target verify: weights read once per iteration; draft is 1/10 the weight.
    target = EntitlementReport(total_flops=1000, total_weight_bytes=1000, total_kv_bytes=200)
    draft = SpeculativeDraft(weight_bytes=100, kv_bytes=20, flops=80)
    gamma = 4
    rep = speculative_report(target, draft, gamma=gamma, acceptance_rate=0.5)

    n = expected_accepted_tokens(gamma, 0.5)  # 1.9375
    iter_weight = 1000 + gamma * 100  # 1400
    iter_kv = 200 + gamma * 20  # 280
    iter_flops = 1000 + gamma * 80  # 1320
    assert rep.expected_accepted == pytest.approx(n)
    assert rep.iter_weight_bytes == iter_weight
    assert rep.iter_kv_bytes == iter_kv
    assert rep.iter_flops == iter_flops
    assert rep.weight_bytes_per_token == pytest.approx(iter_weight / n)
    assert rep.kv_bytes_per_token == pytest.approx(iter_kv / n)
    assert rep.flops_per_token == pytest.approx(iter_flops / n)
    # speedup on the dominant weight-read term = target_weight / (iter_weight / N)
    assert rep.weight_read_speedup_vs_decode == pytest.approx(1000.0 / (iter_weight / n))
    assert rep.draft_weight_fraction == pytest.approx(gamma * 100 / 1000)


def test_speculative_report_accepts_measured_n():
    target = EntitlementReport(total_flops=0, total_weight_bytes=2000, total_kv_bytes=0)
    draft = SpeculativeDraft(weight_bytes=0)
    rep = speculative_report(target, draft, gamma=3, expected_accepted=2.5)
    # zero-cost draft: per-token weight = target_weight / N, speedup = N exactly
    assert rep.weight_bytes_per_token == pytest.approx(2000 / 2.5)
    assert rep.weight_read_speedup_vs_decode == pytest.approx(2.5)


def test_speculative_report_coerces_report_draft():
    target = EntitlementReport(total_flops=0, total_weight_bytes=1000, total_kv_bytes=0)
    draft_report = EntitlementReport(total_flops=50, total_weight_bytes=100, total_kv_bytes=10)
    rep = speculative_report(target, draft_report, gamma=2, acceptance_rate=0.7)
    assert rep.draft_weight_bytes == 100
    assert rep.iter_kv_bytes == 0 + 2 * 10


def test_speculative_report_requires_exactly_one_n_source():
    target = EntitlementReport(total_flops=0, total_weight_bytes=1000, total_kv_bytes=0)
    draft = SpeculativeDraft(weight_bytes=100)
    with pytest.raises(ValueError):
        speculative_report(target, draft, gamma=2)  # neither
    with pytest.raises(ValueError):
        speculative_report(target, draft, gamma=2, acceptance_rate=0.5, expected_accepted=2.0)  # both


def test_speculative_report_rejects_unspecified_draft():
    target = EntitlementReport(total_flops=0, total_weight_bytes=1000, total_kv_bytes=0)
    with pytest.raises(TypeError):
        speculative_report(target, object(), gamma=2, acceptance_rate=0.5)


def test_speculative_report_measured_n_bounds():
    target = EntitlementReport(total_flops=0, total_weight_bytes=1000, total_kv_bytes=0)
    draft = SpeculativeDraft(weight_bytes=0)
    with pytest.raises(ValueError):
        speculative_report(target, draft, gamma=2, expected_accepted=4.0)  # > gamma+1


def test_speculative_draft_fraction_zero_target_safe():
    rep = speculative_report(
        EntitlementReport(total_flops=0, total_weight_bytes=0, total_kv_bytes=0),
        SpeculativeDraft(weight_bytes=0),
        gamma=1,
        expected_accepted=1.0,
    )
    assert rep.draft_weight_fraction == 0.0
    assert rep.weight_read_speedup_vs_decode == 0.0
    assert math.isfinite(rep.arithmetic_intensity)
