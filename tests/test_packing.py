"""Unit tests for sequence packing and document attention masking."""

from __future__ import annotations

import torch

from gslm2.data.text.collators import (
    PackedCausalCollator,
    TextSample,
    _build_packed_tensors,
    _greedy_pack,
)


def _make_sample(ids: list[int]) -> TextSample:
    return TextSample(input_ids=torch.tensor(ids, dtype=torch.long))


class TestGreedyPack:
    def test_basic_two_bins(self) -> None:
        samples = [_make_sample([1, 2, 3, 4, 5]), _make_sample([6, 7, 8]), _make_sample([9, 10, 11, 12, 13, 14])]
        bins = _greedy_pack(samples, max_length=10)
        assert len(bins) == 2
        assert len(bins[0]) == 2  # [5-tok, 3-tok] fit in 10
        assert len(bins[1]) == 1  # [6-tok] starts new bin

    def test_exact_fit(self) -> None:
        samples = [_make_sample([1, 2, 3, 4, 5]), _make_sample([6, 7, 8, 9, 10])]
        bins = _greedy_pack(samples, max_length=10)
        assert len(bins) == 1
        assert len(bins[0]) == 2

    def test_single_oversized_truncated(self) -> None:
        samples = [_make_sample(list(range(20)))]
        bins = _greedy_pack(samples, max_length=8)
        assert len(bins) == 1
        assert bins[0][0].size(0) == 8

    def test_empty_batch(self) -> None:
        bins = _greedy_pack([], max_length=10)
        assert len(bins) == 1
        assert len(bins[0]) == 0

    def test_many_small_samples(self) -> None:
        samples = [_make_sample([i]) for i in range(10)]
        bins = _greedy_pack(samples, max_length=4)
        assert all(sum(doc.size(0) for doc in b) <= 4 for b in bins)
        total_tokens = sum(doc.size(0) for b in bins for doc in b)
        assert total_tokens == 10


class TestBuildPackedTensors:
    def test_shapes_and_values(self) -> None:
        bins = [[torch.tensor([10, 20, 30]), torch.tensor([40, 50])]]
        packed_ids, doc_ids, position_ids = _build_packed_tensors(bins, max_length=7, pad_token_id=0)

        assert packed_ids.shape == (1, 7)
        assert doc_ids.shape == (1, 7)
        assert position_ids.shape == (1, 7)

        assert packed_ids[0].tolist() == [10, 20, 30, 40, 50, 0, 0]
        assert doc_ids[0].tolist() == [0, 0, 0, 1, 1, -1, -1]
        assert position_ids[0].tolist() == [0, 1, 2, 0, 1, 0, 0]

    def test_multi_bin(self) -> None:
        bins = [
            [torch.tensor([1, 2]), torch.tensor([3])],
            [torch.tensor([4, 5, 6])],
        ]
        packed_ids, doc_ids, position_ids = _build_packed_tensors(bins, max_length=4, pad_token_id=99)

        assert packed_ids.shape == (2, 4)
        assert packed_ids[0].tolist() == [1, 2, 3, 99]
        assert packed_ids[1].tolist() == [4, 5, 6, 99]
        assert doc_ids[0].tolist() == [0, 0, 1, -1]
        assert doc_ids[1].tolist() == [0, 0, 0, -1]
        assert position_ids[0].tolist() == [0, 1, 0, 0]
        assert position_ids[1].tolist() == [0, 1, 2, 0]


class TestPackedCausalCollator:
    def test_output_shape_and_type(self) -> None:
        collator = PackedCausalCollator(pad_token_id=0, max_length=10)
        samples = [_make_sample([1, 2, 3]), _make_sample([4, 5]), _make_sample([6, 7, 8, 9])]
        result = collator(samples)
        assert len(result) == 4
        packed_ids, doc_ids, position_ids, targets = result
        assert packed_ids.shape[1] == 10
        assert doc_ids.shape == packed_ids.shape
        assert position_ids.shape == packed_ids.shape
        assert targets.shape == packed_ids.shape

    def test_targets_shifted_within_doc(self) -> None:
        collator = PackedCausalCollator(pad_token_id=0, max_length=8)
        samples = [_make_sample([10, 20, 30]), _make_sample([40, 50])]
        packed_ids, doc_ids, position_ids, targets = collator(samples)

        # Doc 0: [10, 20, 30] -> targets at pos 0,1 should be [20, 30], pos 2 = -100 (boundary)
        assert targets[0, 0].item() == 20
        assert targets[0, 1].item() == 30
        assert targets[0, 2].item() == -100  # last token of doc 0

        # Doc 1: [40, 50] -> target at pos 3 should be 50, pos 4 = -100
        assert targets[0, 3].item() == 50
        assert targets[0, 4].item() == -100  # last token of doc 1

        # Padding positions should be -100
        for i in range(5, 8):
            assert targets[0, i].item() == -100

    def test_single_token_doc(self) -> None:
        collator = PackedCausalCollator(pad_token_id=0, max_length=4)
        samples = [_make_sample([42]), _make_sample([99])]
        packed_ids, doc_ids, position_ids, targets = collator(samples)
        # Single-token docs have no valid targets (length > 1 condition fails)
        assert targets[0, 0].item() == -100
        assert targets[0, 1].item() == -100


class TestPositionIdsReset:
    def test_positions_reset_at_boundary(self) -> None:
        bins = [
            [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6]), torch.tensor([7, 8, 9])],
        ]
        _, _, position_ids = _build_packed_tensors(bins, max_length=12, pad_token_id=0)
        expected = [0, 1, 2, 3, 0, 1, 0, 1, 2, 0, 0, 0]
        assert position_ids[0].tolist() == expected


class TestBlockMaskIsolation:
    def test_causal_mask_isolation(self) -> None:
        from gslm2.models.core.opt import OPT

        doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]])
        assert OPT._build_block_mask(doc_ids) is not None  # production mask constructs

        B, H, Q, KV = 1, 1, 8, 8  # noqa: N806
        mask_tensor = torch.zeros(B, H, Q, KV, dtype=torch.bool)
        for b in range(B):
            for q in range(Q):
                for kv in range(KV):
                    same_doc = doc_ids[b, q] == doc_ids[b, kv]
                    not_pad = doc_ids[b, q] >= 0
                    causal = q >= kv
                    mask_tensor[b, 0, q, kv] = same_doc and not_pad and causal

        # Doc 0 positions (0,1,2): can attend to same doc, causally
        assert mask_tensor[0, 0, 0, 0]  # pos 0 attends to pos 0
        assert mask_tensor[0, 0, 2, 0]  # pos 2 attends to pos 0
        assert not mask_tensor[0, 0, 0, 1]  # pos 0 can't attend to future pos 1 (causal)

        # Cross-doc attention blocked
        assert not mask_tensor[0, 0, 3, 2]  # doc 1 pos 3 can't attend to doc 0 pos 2
        assert not mask_tensor[0, 0, 2, 3]  # doc 0 pos 2 can't attend to doc 1 pos 3

        # Padding positions can't attend or be attended to
        assert not mask_tensor[0, 0, 6, 0]  # pad can't attend to doc 0
        assert not mask_tensor[0, 0, 0, 6]  # doc 0 can't attend to pad


class TestModelRoundtrip:
    def test_opt_packed_forward(self) -> None:
        from gslm2.config.models.core import CausalLMConfig
        from gslm2.models.core.opt import OPT

        cfg = CausalLMConfig(
            model_dim=64,
            max_seq_len=32,
            vocab_size=100,
            num_layers=2,
            num_attn_heads=4,
            num_key_value_heads=4,
            ffn_inner_dim=128,
            dropout_p=0.0,
        )
        model = OPT(cfg)
        model.train()

        collator = PackedCausalCollator(pad_token_id=1, max_length=32)
        samples = [
            _make_sample(list(range(2, 12))),
            _make_sample(list(range(12, 20))),
            _make_sample(list(range(20, 28))),
        ]
        packed_ids, doc_ids, position_ids, targets = collator(samples)

        loss, metrics = model(packed_ids, targets=targets, position_ids=position_ids, doc_ids=doc_ids)
        assert loss.isfinite()
        assert loss.item() > 0
        loss.backward()

    def test_opt_nonpacked_still_works(self) -> None:
        from gslm2.config.models.core import CausalLMConfig
        from gslm2.data.text.collators import CausalCollator
        from gslm2.models.core.opt import OPT

        cfg = CausalLMConfig(
            model_dim=64,
            max_seq_len=32,
            vocab_size=100,
            num_layers=2,
            num_attn_heads=4,
            num_key_value_heads=4,
            ffn_inner_dim=128,
            dropout_p=0.0,
        )
        model = OPT(cfg)
        model.train()

        collator = CausalCollator(pad_token_id=1)
        samples = [_make_sample(list(range(2, 12))), _make_sample(list(range(12, 20)))]
        ids, attn, targets = collator(samples)

        loss, metrics = model(ids, targets=targets, attention_mask=attn)
        assert loss.isfinite()
        loss.backward()

    def test_opt_all_single_token_no_nan(self) -> None:
        """A packed bin of only single-token docs has zero supervised targets;
        loss must be finite (not NaN) and still backprop."""
        from gslm2.config.models.core import CausalLMConfig
        from gslm2.models.core.opt import OPT

        cfg = CausalLMConfig(
            model_dim=64,
            max_seq_len=32,
            vocab_size=100,
            num_layers=2,
            num_attn_heads=4,
            num_key_value_heads=4,
            ffn_inner_dim=128,
            dropout_p=0.0,
        )
        model = OPT(cfg).train()
        collator = PackedCausalCollator(pad_token_id=1, max_length=32)
        samples = [_make_sample([5]), _make_sample([7])]  # single-token docs → all targets ignored
        packed_ids, doc_ids, position_ids, targets = collator(samples)
        assert (targets == -100).all()  # precondition: no supervised targets in this batch
        loss, _ = model(packed_ids, targets=targets, position_ids=position_ids, doc_ids=doc_ids)
        assert loss.isfinite()
        loss.backward()

    def test_packed_loss_decreases(self) -> None:
        """Train a few steps with packing and verify loss decreases."""
        from torch.optim import AdamW

        from gslm2.config.models.core import CausalLMConfig
        from gslm2.models.core.opt import OPT

        torch.manual_seed(42)
        cfg = CausalLMConfig(
            model_dim=64,
            max_seq_len=32,
            vocab_size=100,
            num_layers=2,
            num_attn_heads=4,
            num_key_value_heads=4,
            ffn_inner_dim=128,
            dropout_p=0.0,
        )
        model = OPT(cfg)
        model.train()
        optimizer = AdamW(model.parameters(), lr=1e-3)

        collator = PackedCausalCollator(pad_token_id=1, max_length=32)
        samples = [_make_sample(list(range(2, 10))) for _ in range(8)]

        losses = []
        for _ in range(10):
            packed_ids, doc_ids, position_ids, targets = collator(samples)
            loss, _ = model(packed_ids, targets=targets, position_ids=position_ids, doc_ids=doc_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestPackedUnpackedEquivalence:
    """The core packing invariant: a document packed (alone or with others) must
    produce the *same* logits as running that document unpacked. This proves the
    BlockMask blocks cross-document attention, position-ids reset correctly, and
    GQA behaves identically on the flex (packed) and SDPA (unpacked) paths.
    """

    @staticmethod
    def _opt(num_kv_heads: int = 4):  # noqa: ANN205
        from gslm2.config.models.core import CausalLMConfig
        from gslm2.models.core.opt import OPT

        torch.manual_seed(0)
        cfg = CausalLMConfig(
            model_dim=64,
            max_seq_len=32,
            vocab_size=100,
            num_layers=2,
            num_attn_heads=4,
            num_key_value_heads=num_kv_heads,
            ffn_inner_dim=128,
            dropout_p=0.0,
        )
        return OPT(cfg).eval()

    def _packed_logits(self, model, doc_lists: list[list[int]], max_length: int = 32):
        collator = PackedCausalCollator(pad_token_id=1, max_length=max_length)
        packed_ids, doc_ids, position_ids, _ = collator([_make_sample(d) for d in doc_lists])
        with torch.no_grad():
            return model(packed_ids, position_ids=position_ids, doc_ids=doc_ids), doc_ids

    def test_opt_single_doc_equivalence(self) -> None:
        model = self._opt()
        doc = list(range(2, 14))  # 12 tokens
        with torch.no_grad():
            plain = model(torch.tensor([doc], dtype=torch.long))  # unpacked causal forward
        packed, _ = self._packed_logits(model, [doc])
        torch.testing.assert_close(packed[0, : len(doc)], plain[0], atol=1e-4, rtol=1e-3)

    def test_dense_and_flex_block_masks_agree(self) -> None:
        """Packed training routes through SDPA with a dense block mask (which honours dropout_p);
        eval uses the flex BlockMask. At dropout=0 the two paths must produce identical logits on
        supervised positions, so switching between them across train/eval does not change the
        function.
        """
        model = self._opt()  # dropout=0
        doc_a, doc_b = list(range(2, 12)), list(range(20, 27))
        collator = PackedCausalCollator(pad_token_id=1, max_length=32)
        packed_ids, doc_ids, position_ids, _ = collator([_make_sample(doc_a), _make_sample(doc_b)])
        with torch.no_grad():
            flex = model.eval()(packed_ids, position_ids=position_ids, doc_ids=doc_ids)
            dense = model.train()(packed_ids, position_ids=position_ids, doc_ids=doc_ids)
        n = len(doc_a) + len(doc_b)
        torch.testing.assert_close(flex[0, :n], dense[0, :n], atol=1e-4, rtol=1e-3)

    def test_dense_blockmask_path_applies_attention_dropout(self) -> None:
        """The packed TRAINING path must apply attention dropout, like the unpacked (SDPA) path.
        flex_attention takes no dropout argument, so it can silently omit it, under-regularising
        packed training; training therefore routes through SDPA with a dense block mask. Tested at
        the attention-module level (dropout=0.5) so the block's other dropouts don't mask the
        difference: two training forwards must differ, two eval forwards must match.
        """
        from gslm2.models.core.opt import OPT
        from gslm2.models.modules.transformer import MultiheadSelfAttention

        torch.manual_seed(0)
        attn = MultiheadSelfAttention(64, 4, 4, dropout=0.5, causal=True, attn_window_len=None).train()
        x = torch.randn(1, 12, 64)
        dense = OPT._build_block_mask_dense(torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]))
        with torch.no_grad():
            y1 = attn(x, attention_mask=dense)
            y2 = attn(x, attention_mask=dense)
        assert not torch.allclose(y1, y2), "packed/dense path applied no attention dropout in training mode"
        attn.eval()
        with torch.no_grad():
            torch.testing.assert_close(attn(x, attention_mask=dense), attn(x, attention_mask=dense))

    def test_packed_training_no_nan_with_padding(self) -> None:
        """A padding-query row (doc_id < 0) is fully masked; SDPA emits NaN for an empty row
        (flex_attention returns 0), which would poison gradients via NaN*0. The dense block mask
        keeps every padding row attending to itself, so a packed training forward AND backward over
        a bin with trailing padding must stay finite.
        """
        from gslm2.config.models.core import CausalLMConfig
        from gslm2.models.core.opt import OPT

        torch.manual_seed(0)
        cfg = CausalLMConfig(
            model_dim=64,
            max_seq_len=32,
            vocab_size=100,
            num_layers=2,
            num_attn_heads=4,
            num_key_value_heads=4,
            ffn_inner_dim=128,
            dropout_p=0.5,
        )
        model = OPT(cfg).train()
        collator = PackedCausalCollator(pad_token_id=1, max_length=32)  # two short docs leave padding
        packed_ids, doc_ids, position_ids, targets = collator(
            [_make_sample(list(range(2, 10))), _make_sample(list(range(10, 16)))]
        )
        assert (doc_ids < 0).any(), "test needs trailing padding to exercise the NaN guard"
        loss, _ = model(packed_ids, targets=targets, position_ids=position_ids, doc_ids=doc_ids)
        loss.backward()
        assert torch.isfinite(loss), "packed training loss is not finite (padding-row NaN?)"
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "no gradients produced"
        assert all(torch.isfinite(g).all() for g in grads), "non-finite grads (padding-row NaN?)"

    def test_opt_two_docs_isolation(self) -> None:
        """Two docs packed together: the SECOND doc's logits must equal the doc run
        alone — i.e. it does not attend across the boundary to the first doc."""
        model = self._opt()
        doc_a = list(range(2, 12))  # 10 tokens
        doc_b = list(range(20, 27))  # 7 tokens
        with torch.no_grad():
            plain_a = model(torch.tensor([doc_a], dtype=torch.long))
            plain_b = model(torch.tensor([doc_b], dtype=torch.long))
        packed, doc_ids = self._packed_logits(model, [doc_a, doc_b])
        assert doc_ids[0, : len(doc_a)].eq(0).all()
        assert doc_ids[0, len(doc_a) : len(doc_a) + len(doc_b)].eq(1).all()
        torch.testing.assert_close(packed[0, : len(doc_a)], plain_a[0], atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(packed[0, len(doc_a) : len(doc_a) + len(doc_b)], plain_b[0], atol=1e-4, rtol=1e-3)

    def test_opt_gqa_equivalence(self) -> None:
        """GQA: packed path uses flex_attention(enable_gqa=True); unpacked uses SDPA
        with repeat_interleave. They must agree."""
        model = self._opt(num_kv_heads=2)  # group_size = 2
        doc = list(range(2, 15))
        with torch.no_grad():
            plain = model(torch.tensor([doc], dtype=torch.long))
        packed, _ = self._packed_logits(model, [doc])
        torch.testing.assert_close(packed[0, : len(doc)], plain[0], atol=1e-4, rtol=1e-3)
