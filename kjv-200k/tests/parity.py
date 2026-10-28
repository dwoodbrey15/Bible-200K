"""Emit a temporary PyTorch fixture for the Node parity test."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Config, Transformer
from convert import to_browser
torch.set_num_threads(2)
torch.manual_seed(42)
m = Transformer().eval()
ids = torch.tensor([list(b"In the beginning")])
packed = to_browser({"model": m.state_dict(), "config": asdict(m.cfg), "manifest": {}})
Path(sys.argv[1]).write_text(json.dumps({"checkpoint": packed, "ids": ids.tolist(), "logits": m(ids)[0].flatten().tolist()}))
