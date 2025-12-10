from datasets import load_dataset

print("Loading yerevann/coco-karpathy...")
ds = load_dataset("yerevann/coco-karpathy", split="test")
print("Column names:", ds.column_names)
print("First item keys:", ds[0].keys())
print("First item content (partial):", {k: str(v)[:50] for k, v in ds[0].items()})
