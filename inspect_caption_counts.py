from datasets import load_dataset
import ast

def get_all_captions(item, col_name):
    val = item.get(col_name, [])
    if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
        try:
            val = ast.literal_eval(val)
        except:
            pass
    if not isinstance(val, list):
        val = [str(val)]
    return [str(v) for v in val]

print("Loading COCO Karpathy Test Split...")
ds = load_dataset("yerevann/coco-karpathy", split="test")

counts = {}
for idx, item in enumerate(ds):
    caps = get_all_captions(item, "sentences")
    l = len(caps)
    counts[l] = counts.get(l, 0) + 1
    if l != 5 and counts[l] <= 3:
        print(f"Example with {l} captions (idx {idx}): {caps}")

print("\nCaption Count Distribution:")
for l, count in sorted(counts.items()):
    print(f"Length {l}: {count} images")
