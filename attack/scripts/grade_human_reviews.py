import pandas as pd

def analyze_quality_preservation(csv_path):
    """
    Reads a CSV file and analyzes quality preservation based on human evaluation.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        
    Returns:
        pandas.DataFrame: A summary table showing quality preservation breakdown by mutator and watermark.
    """
    import pandas as pd

    # Load dataset
    df = pd.read_csv(csv_path)

    # Ensure relevant columns exist
    required_columns = ["mutated_text", "Which response is better? A, B, or TIE", "text_A", "text_B", "mutator", "watermark"]
    df = df[required_columns].dropna()

    # Function to classify quality preservation
    def classify_quality(row):
        if row["mutated_text"] == "text_A":
            if row["Which response is better? A, B, or TIE"] in ["A", "TIE"]:
                return "quality_preserved"
            elif row["Which response is better? A, B, or TIE"] == "B":
                return "quality_not_preserved"
        elif row["mutated_text"] == "text_B":
            if row["Which response is better? A, B, or TIE"] in ["B", "TIE"]:
                return "quality_preserved"
            elif row["Which response is better? A, B, or TIE"] == "A":
                return "quality_not_preserved"
        return None

    # Apply classification
    df["quality"] = df.apply(classify_quality, axis=1)

    # Aggregate results
    summary = df.groupby(["mutator", "watermark"]).agg(
        total_cases=("mutated_text", "count"),
        quality_preserved=("quality", lambda x: (x == "quality_preserved").sum()),
        quality_not_preserved=("quality", lambda x: (x == "quality_not_preserved").sum())
    ).reset_index()

    # Compute percentages
    summary["quality_preserved_percentage"] = (summary["quality_preserved"] / summary["total_cases"]) * 100
    summary["quality_not_preserved_percentage"] = (summary["quality_not_preserved"] / summary["total_cases"]) * 100

    return summary

def main():
    file_path = "./data/final_review/RQ3_Final_Human_Review_on_Broken_N_=20_per_Mutator+Watermarker - subsampled_dataset_len=288_obscured.csv"
    df = analyze_quality_preservation(file_path)
    print(df)
    grade_path = file_path.replace(".csv", "_graded.csv")
    df.to_csv(grade_path)
    print(f"Graded human review saved to: {grade_path}")

if __name__ == "__main__":

    # python -m attack.scripts.grade_human_reviews
    
    main()