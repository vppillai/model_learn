"""Lab 05 — how temperature and top_k reshape the next-token distribution.
Run: PYTHONPATH=src python labs/lab05_sampling.py
"""
import torch
torch.manual_seed(0)
logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0, -1.0])
for temp in (0.2, 0.8, 1.5):
    p = torch.softmax(logits / temp, dim=-1)
    print(f"temp={temp}: " + " ".join(f"{x:.2f}" for x in p))
print("\nLow temp -> peaky (greedy, repetitive). High temp -> flat (creative,")
print("risky). top_k just zeroes everything outside the k most likely tokens.")
