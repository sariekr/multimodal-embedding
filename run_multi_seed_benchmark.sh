#!/bin/bash
# Multi-seed benchmark runner for statistical significance
# Runs v18 benchmark with 5 different seeds and aggregates results

SEEDS=(42 123 456 789 1011)
OUTPUT_DIR="results_multiseed"
mkdir -p $OUTPUT_DIR

echo "=================================="
echo "MULTI-SEED BENCHMARK"
echo "Running ${#SEEDS[@]} iterations"
echo "=================================="

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    RUN_NUM=$((i + 1))

    echo ""
    echo "##################################################"
    echo "### RUN $RUN_NUM/${#SEEDS[@]} - SEED=$SEED"
    echo "##################################################"

    # Modify Python script to use this seed
    sed "s/^SEED = [0-9]*/SEED = $SEED/" run_benchmark_grand_slam_v18.py > /tmp/run_temp_${SEED}.py

    # Run benchmark
    python /tmp/run_temp_${SEED}.py

    # Move results
    mv benchmark_v18_results.csv "$OUTPUT_DIR/results_seed_${SEED}.csv"

    echo "✓ Run $RUN_NUM complete"
done

echo ""
echo "=================================="
echo "Aggregating results..."
echo "=================================="

# Run Python aggregation script
python - <<'EOF'
import pandas as pd
import numpy as np
from pathlib import Path

results_dir = Path("results_multiseed")
all_dfs = []

for csv_file in sorted(results_dir.glob("results_seed_*.csv")):
    df = pd.read_csv(csv_file)
    all_dfs.append(df)

# Aggregate by model
models = all_dfs[0]["Model"].tolist()
aggregated = []

for model in models:
    row = {"Model": model}

    # Get all runs for this model
    model_runs = [df[df["Model"] == model] for df in all_dfs]

    # For each metric column
    for col in all_dfs[0].columns:
        if col == "Model":
            continue

        values = []
        for run_df in model_runs:
            val = run_df[col].values[0] if len(run_df) > 0 else None

            # Parse percentage strings
            if isinstance(val, str) and "%" in val:
                try:
                    val_float = float(val.strip("%"))
                    values.append(val_float)
                except:
                    pass
            elif isinstance(val, (int, float)):
                values.append(val)

        if len(values) > 0:
            mean = np.mean(values)
            std = np.std(values, ddof=1)  # Sample std
            row[col] = f"{mean:.1f}±{std:.1f}%"
        else:
            row[col] = "N/A"

    aggregated.append(row)

# Save aggregated results
df_agg = pd.DataFrame(aggregated)
df_agg.to_csv("benchmark_v18_multiseed_aggregated.csv", index=False)

print("✅ Aggregated results saved to: benchmark_v18_multiseed_aggregated.csv")
print(df_agg.to_markdown(index=False))
EOF

echo "=================================="
echo "MULTI-SEED BENCHMARK COMPLETE"
echo "=================================="
