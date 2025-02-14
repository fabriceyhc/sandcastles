from attack.utils import load_all_csvs

if __name__ == "__main__":
    watermarks = ["Adaptive", "KGW", "SIR"]
    mutators = [
        "DocumentMutator",
        "Document1StepMutator",
        "Document2StepMutator",
        "SentenceMutator",
        "SpanMutator",
        "WordMutator",
        "EntropyWordMutator",
    ]
    for idm, mutator in enumerate(mutators):
        groups = 0
        total = 0
        # print(f"Analyzing {watermarker}")
        
        for idx, watermarker in enumerate(watermarks):
            # print(f"Analyzing for {mutator}")
            df_total = load_all_csvs("./attack/traces/annotated", watermarker, mutator)
            if df_total.empty:
                continue

            for id, group in df_total.groupby("group_id"):
                assert len(group[group['quality_preserved'] == True]) + len(group[group['quality_preserved'] == False]) == len(group)
                group = group[group['mutation_num'] >= 0]
                if len(group[group['quality_preserved'].astype(bool) == True]) == 0:
                    total += 1
                groups += 1
        print(f"{mutator} had {total}/{groups} groups never have a successful mutation")