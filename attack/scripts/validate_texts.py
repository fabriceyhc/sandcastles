import pandas as pd
from attack.utils import load_all_csvs

watermark_types = ["Adaptive", "KGW", "SIR","GPT4o_unwatermarked"]
mutators = [
    "DocumentMutator", "Document1StepMutator", "Document2StepMutator",
    "SentenceMutator", "SpanMutator", "WordMutator", "EntropyWordMutator"
]

results = []

for watermark_type in watermark_types:
    for mutator in mutators:
        # Load data with fallback to non-annotated directory
        df = load_all_csvs("./attack/traces/annotated", watermark_type, mutator)
        if df.empty:
            df = load_all_csvs("./attack/traces", watermark_type, mutator)
        
        if df.empty:
            print(f"[MAIN] No data for {watermark_type} + {mutator}")
            results.append({
                "watermark_type": watermark_type,
                "mutator": mutator,
                "violation_count_before": 0,
                "violation_count_after": 0
            })
            continue
        
        # Clean boolean column
        df['quality_preserved'] = df['quality_preserved'].astype(str).str.lower().map({'true': True, 'false': False})
        
        violation_before, violation_after = 0, 0
        
        for _, group in df.groupby("group_id"):
            group = group.sort_values("step_num").reset_index(drop=True)
            if len(group) < 2:
                continue
            
            # Create mask: only check rows where:
            # 1. Current row has quality_preserved=True
            # 2. Current row isn't the first row (step_num != -1)
            # 3. Current row isn't the last row (has a next row)
            check_mask = (
                group['quality_preserved'] &
                (group['step_num'] != -1) &
                (group.index < len(group)-1)
            )
            
            # Get next row's current_text for comparison
            next_current = group['current_text'].shift(-1)
            
            # BEFORE FIX: Check violations in original data
            violation_before += (
                (group['mutated_text'] != next_current) & check_mask
            ).sum()
            
            # Apply fix: Propagate mutated_text forward where quality preserved
            fixed_current = group['current_text'].copy()
            for i in range(1, len(group)):
                if group.at[i-1, 'quality_preserved'] and group.at[i-1, 'step_num'] != -1:
                    fixed_current.at[i] = group.at[i-1, 'mutated_text']
            
            # AFTER FIX: Check violations with corrected current_text
            next_fixed = fixed_current.shift(-1)
            violation_after += (
                (group['mutated_text'] != next_fixed) & check_mask
            ).sum()


        print(f"[CHECK] {watermark_type} + {mutator} → Violations Before: {violation_before}, After: {violation_after}")
        results.append({
            "watermark_type": watermark_type,
            "mutator": mutator,
            "violation_count_before": violation_before,
            "violation_count_after": violation_after
        })

df_results = pd.DataFrame(results)
print(df_results)