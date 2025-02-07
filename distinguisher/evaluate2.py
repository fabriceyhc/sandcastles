# ./impossibility-watermark> python -m distinguisher.evaluate2

from distinguisher.models import (SimplestGPT)
from distinguisher.utils import get_model
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def evaluate2(distinguisher, rows):
    """
    Given a distinguisher and a list of rows, evaluate the distinguisher on each row.
    Rows should contain at minimum the following parts: P, origin_A, origin_B, Origin
    P is the perturbed response that should be distinguished against origin_A and origin_B.
    Origin is the origin of the perturbed response, used to validate the distinguisher.
    """
    for i, row in rows.iterrows():
        log.info(f"Processing row {i+1}/{len(rows)}")
        distinguisher.set_origin(row['origin_A'], row['origin_B'])
        output = distinguisher.distinguish_row(row, prefix="2nd_")
        for key, value in output.items():
            rows.at[i, key] = value
    return rows

def main():
    # Load the distinguisher
    llm, distinguisher_persona  = get_model("gpt-4o-mini")
    distinguisher = SimplestGPT(llm, distinguisher_persona)
    # Load the dataset
    df = pd.read_csv("distinguisher/tryagain.csv")
    df = evaluate2(distinguisher, df)
    df.to_csv("distinguisher/results/redemption.csv", index=False)

if __name__ == "__main__":
    main()