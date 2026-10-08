import unittest

import torch
from torch.nn.utils.rnn import pad_sequence

from gslm2.data.common.masking.speech import build_attention_mask
from gslm2.models.speech.components import ConvPositionalEmbedding


class TestConvPosEmbBatching(unittest.TestCase):
    def setUp(self) -> None:
        self.seq = 100
        self.emb_dim = 768
        self.n_padding = 5
        self.x = torch.randn((self.seq, self.emb_dim))
        self.y = torch.randn((self.seq + self.n_padding, self.emb_dim))
        self.lengths = torch.tensor([self.seq, self.seq + self.n_padding])
        self.batch = pad_sequence([self.x, self.y], batch_first=True)
        self.module = ConvPositionalEmbedding(embed_dim=768, kernel_size=95, groups=16, depth=5)

    @torch.no_grad()
    def test_without_attention_mask(self) -> None:
        pos_x = self.module(self.x.unsqueeze(0)).squeeze(0)
        pos_y = self.module(self.y.unsqueeze(0)).squeeze(0)
        pos_batch = self.module(self.batch)
        with self.assertRaises(AssertionError) as context:
            torch.testing.assert_close(pos_x, pos_batch[0, : self.seq])
        self.assertIn("Mismatched elements:", str(context.exception))
        torch.testing.assert_close(pos_y, pos_batch[1])

    @torch.no_grad()
    def test_with_attention_mask(self) -> None:
        pos_x = self.module(self.x.unsqueeze(0)).squeeze(0)
        pos_y = self.module(self.y.unsqueeze(0)).squeeze(0)
        pos_batch = self.module(self.batch, build_attention_mask(self.lengths))
        torch.testing.assert_close(pos_x, pos_batch[0, : self.seq])
        torch.testing.assert_close(pos_y, pos_batch[1])


if __name__ == "__main__":
    unittest.main()
