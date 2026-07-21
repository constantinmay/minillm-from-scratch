"""Interactive generation - type your own prompts!"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from model.config import MiniLLMConfig
from model.gpt import MiniLLM
from model.generation import generate
from tokenizer.tokenizer_utils import MiniLLMTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_config = MiniLLMConfig.from_yaml("configs/model_config.yaml")
tokenizer = MiniLLMTokenizer("tokenizer/minillm_tokenizer.json")

model = MiniLLM(model_config).to(device)
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/base.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
print(f"Loaded step {ckpt.get('step', 0)}, {model.num_params():,} params")
model.eval()

print("\n=== MiniLLM Interactive Generator ===")
print("Type a prompt and press Enter. Type 'quit' to exit.\n")

sft_mode = "--sft" in sys.argv or len(sys.argv) > 2
while True:
    prompt = input("Instruction> ").strip() if sft_mode else input("Prompt> ").strip()
    if prompt.lower() in ("quit", "exit", "q"):
        break
    if not prompt:
        continue
    if sft_mode:
        prompt = f"### Instruction:\n{prompt}\n### Response:\n"
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.long, device=device,
    )
    output_ids = generate(
        model, input_ids,
        max_new_tokens=128,
        temperature=0.8,
        top_k=40,
        eos_token_id=tokenizer.eos_id(),
    )
    text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    print(f"\n{text}\n")
