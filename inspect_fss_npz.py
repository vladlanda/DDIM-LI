"""Quick inspector: print the actual shape and values of FSS arrays in an npz."""
import argparse, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--npz", required=True)
args = p.parse_args()

data = np.load(args.npz, allow_pickle=True)
print("All keys containing 'fss':")
fss_keys = [k for k in data.files if "fss" in k.lower()]
for k in sorted(fss_keys):
    arr = data[k]
    print(f"  {k:24s} shape={arr.shape} dtype={arr.dtype}")
    if arr.ndim >= 1 and arr.size <= 20:
        print(f"      values = {np.round(arr, 4)}")

print()
print("fss_scales:", data["fss_scales"].tolist() if "fss_scales" in data else "MISSING")
print("fss_prob_thresholds:", data["fss_prob_thresholds"].tolist() if "fss_prob_thresholds" in data else "MISSING")
print("pr_steps:", data["pr_steps"].tolist() if "pr_steps" in data else "MISSING")
