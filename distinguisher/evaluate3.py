# ./impossibility-watermark> python -m distinguisher.evaluate3

from distinguisher.models import (SimplestGPT, LogicGPT)
from distinguisher.utils import get_model
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def evaluate3(distinguisher, rows, output_path, skip=0):
    """
    Given a distinguisher and a list of rows, evaluate the distinguisher on each row.
    Rows should contain at minimum the following parts: P, origin_A, origin_B, Origin
    P is the perturbed response that should be distinguished against origin_A and origin_B.
    Origin is the origin of the perturbed response, used to validate the distinguisher.
    If the distinguisher crashes on row X/T, set skip=X-1 to skip the first X-1 rows.
    """
    for i, row in rows.iterrows():
        if i < skip:
            continue
        log.info(f"Processing row {i+1}/{len(rows)}")
        distinguisher.set_origin(row['origin_A'], row['origin_B'])
        output = distinguisher.distinguish_row(row, prefix="3rd_")
        for key, value in output.items():
            rows.at[i, key] = value
        # append row to the file
        pd.DataFrame([rows.iloc[i]]).to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

def main():
    # Load the distinguisher
    llm, distinguisher_persona  = get_model("o1-mini")
    distinguisher = SimplestGPT(llm, distinguisher_persona)
    # Load the dataset
    df = pd.read_csv("distinguisher/data/failed_redemption_long_with_4o_logic.csv")
    evaluate3(distinguisher, df, "distinguisher/results/redemption_full_with_o1.csv")

if __name__ == "__main__":
    main()