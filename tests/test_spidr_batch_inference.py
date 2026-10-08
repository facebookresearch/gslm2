# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.nn.utils.rnn import pad_sequence

from gslm2.config import SpidRConfig
from gslm2.data.common.masking.speech import build_attention_mask
from gslm2.data.speech.collators import conv_length
from gslm2.models import build_model


class TestSpidRBatching(unittest.TestCase):
    def setUp(self) -> None:
        self.seq = 10000
        self.n_padding = 6000
        self.model_cfg = SpidRConfig()
        self.model = build_model(cfg=self.model_cfg, model_type="spidr").eval()
        self.x = torch.randn(self.seq)
        self.y = torch.randn(self.seq + self.n_padding)
        self.lengths = torch.tensor([self.seq, self.seq + self.n_padding])
        self.batch = pad_sequence([self.x, self.y], batch_first=True)

    def tearDown(self) -> None:
        """Clean up model and tensors after each test."""
        del self.model
        del self.model_cfg
        del self.x, self.y, self.lengths, self.batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def test_codebooks_without_attention_mask(self) -> None:
        codebooks_x = self.model.get_codebooks(self.x.unsqueeze(0))
        codebooks_y = self.model.get_codebooks(self.y.unsqueeze(0))
        codebooks_batch = self.model.get_codebooks(self.batch)
        for layer in range(5, 13):
            with self.assertRaises(AssertionError) as context:
                torch.testing.assert_close(
                    codebooks_x[layer - 1].squeeze(0),
                    codebooks_batch[layer - 1][0, : codebooks_x[layer - 1].shape[0]],
                )
            self.assertIn("Mismatched elements:", str(context.exception))
            torch.testing.assert_close(
                codebooks_y[layer - 1].squeeze(0),
                codebooks_batch[layer - 1][1, : codebooks_y[layer - 1].shape[0]],
            )

        # Clean up test variables
        del codebooks_x, codebooks_y, codebooks_batch

    def test_codebooks_with_attention_mask(self) -> None:
        conv_lengths = conv_length(self.model_cfg.extractor_conv_layer_config, self.lengths)
        attention_mask = build_attention_mask(conv_lengths)
        codebooks_x = self.model.get_codebooks(self.x.unsqueeze(0))
        codebooks_y = self.model.get_codebooks(self.y.unsqueeze(0))
        for layer in range(5, 13):
            codebooks_batch = self.model.get_codebooks(self.batch, attention_mask=attention_mask, layers=[layer])
            torch.testing.assert_close(
                codebooks_x[layer - 1].squeeze(0),
                codebooks_batch[layer - 1][0, : conv_lengths[0]],
            )
            torch.testing.assert_close(
                codebooks_y[layer - 1].squeeze(0),
                codebooks_batch[layer - 1][1, : conv_lengths[1]],
            )
            for other_layer in range(1, 13):
                if other_layer != layer:
                    self.assertIsNone(codebooks_batch[other_layer - 1])

        # Clean up test variables
        del conv_lengths, attention_mask, codebooks_x, codebooks_y, codebooks_batch

    def test_intermediate_outputs_with_attention_mask(self) -> None:
        conv_lengths = conv_length(self.model_cfg.extractor_conv_layer_config, self.lengths)
        attention_mask = build_attention_mask(conv_lengths)
        codebooks_x = self.model.get_intermediate_outputs(self.x.unsqueeze(0))
        codebooks_y = self.model.get_intermediate_outputs(self.y.unsqueeze(0))
        for layer in range(1, 13):
            codebooks_batch = self.model.get_intermediate_outputs(
                self.batch, attention_mask=attention_mask, layers=[layer]
            )
            torch.testing.assert_close(
                codebooks_x[layer - 1].squeeze(0),
                codebooks_batch[layer - 1][0, : conv_lengths[0]],
            )
            torch.testing.assert_close(
                codebooks_y[layer - 1].squeeze(0),
                codebooks_batch[layer - 1][1, : conv_lengths[1]],
            )
            for other_layer in range(1, 13):
                if other_layer != layer:
                    self.assertIsNone(codebooks_batch[other_layer - 1])

        # Clean up test variables
        del conv_lengths, attention_mask, codebooks_x, codebooks_y, codebooks_batch


if __name__ == "__main__":
    unittest.main()
