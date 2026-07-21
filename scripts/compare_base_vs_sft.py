"""Compare v3 base vs SFT checkpoints."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from model.config import MiniLLMConfig
from model.gpt import MiniLLM
from model.generation import generate
from tokenizer.tokenizer_utils import MiniLLMTokenizer

device = "cuda"
tok = MiniLLMTokenizer("tokenizer/minillm_tokenizer.json")
model_config = MiniLLMConfig.from_yaml("configs/model_config.yaml")

prompts = [
    "Once upon a time",
    "The cat",
    "Once upon a time, there was a little girl named",
    "A dog and a cat went to the park",
    "The old man walked into the forest",
    "One day, a little bird fell from the tree",
]

checkpoints = [
    ("BASE", "checkpoints_v3/base.pt"),
    ("SFT-50", "checkpoints_v3/sft_step_50.pt"),
    ("SFT-100", "checkpoints_v3/sft_step_100.pt"),
    ("SFT-150", "checkpoints_v3/sft_step_150.pt"),
    ("SFT-200", "checkpoints_v3/sft_step_200.pt"),
    ("SFT-final", "checkpoints_v3/sft.pt"),
]

model = MiniLLM(model_config).to(device)

for name, path in checkpoints:
    if not os.path.exists(path):
        continue
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"\n{'='*60}")
    print(f"  {name} (step {ckpt.get('step', '?')})")
    print(f"{'='*60}")
    for p in prompts:
        ids = torch.tensor([tok.encode(p + " ", add_special_tokens=False)], dtype=torch.long, device=device)
        out = generate(model, ids, max_new_tokens=80, temperature=0.8, top_k=40, eos_token_id=tok.eos_id())
        text = tok.decode(out[0].tolist(), skip_special_tokens=True)
        gen = text[len(p)+1:] if text.startswith(p) else text
        print(f"  [{p[:35]:>35}] -> {gen[:100]}")
