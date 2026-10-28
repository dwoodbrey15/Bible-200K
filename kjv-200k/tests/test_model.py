import hashlib
import json
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Config, Transformer
from convert import to_browser, to_torch

torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def test_size_and_backward(self):
        m = Transformer()
        self.assertEqual(m.parameter_count(), 205376)
        _, loss = m(torch.randint(256, (2, 16)), torch.randint(256, (2, 16)))
        loss.backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters()))

    def test_causality(self):
        m = Transformer(Config(dim=32, layers=2, heads=4, hidden=64, context=32)).eval()
        a = torch.randint(256, (1, 24))
        b = a.clone()
        b[:, 12:] = torch.randint(256, (1, 12))
        torch.testing.assert_close(m(a)[0][:, :12], m(b)[0][:, :12], atol=1e-6, rtol=1e-5)

    def test_learning_and_weight_conversion(self):
        cfg = Config(dim=32, layers=2, heads=4, hidden=64, context=32)
        m = Transformer(cfg)
        data = list(b"In the beginning God created the heaven and the earth. " * 2)
        x = torch.tensor([data[:32]])
        y = torch.tensor([data[1:33]])
        opt = torch.optim.AdamW(m.parameters(), lr=0.003)
        start = m(x, y)[1].item()
        for _ in range(25):
            opt.zero_grad()
            m(x, y)[1].backward()
            opt.step()
        self.assertLess(m(x, y)[1].item(), start - 1)
        from dataclasses import asdict
        packed = to_browser({"model": m.state_dict(), "config": asdict(cfg), "manifest": {}, "step": 25})
        restored = to_torch(packed)
        for key, value in m.state_dict().items():
            self.assertTrue(torch.equal(value, restored["model"][key]))

    def test_corpus_checksums(self):
        root = Path(__file__).resolve().parents[1] / "data"
        manifest = json.loads((root / "manifest.json").read_text())
        for name in ("train", "val"):
            data = (root / f"{name}.bin").read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), manifest["streams"][name]["sha256"])
            self.assertEqual(len(data), manifest["streams"][name]["bytes"])
            self.assertNotIn(b"<", data)
        self.assertEqual(len(manifest["validation_chapters"]), 59)


if __name__ == "__main__":
    unittest.main()
