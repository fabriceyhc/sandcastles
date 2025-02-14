import pandas as pd
import numpy as np
import language_tool_python
from tqdm import tqdm
tqdm.pandas()  # Initialize tqdm for pandas

class GrammarMetric:
    def __init__(self) -> None:
        """
        Use language_tool_python to check grammar.
        :Package Requirements:
            * pip install language_tool_python
        :Language: english
        """
        self.language_tool = language_tool_python.LanguageTool('en-US')

    def find_grammar_issues(self, text, early=False):
        # Check for null/NaN or non-string values
        if text is None or not isinstance(text, str):
            return []
        
        # Optionally limit to 50 words if early==True
        if early:
            text = ' '.join(text.split()[:50])
        
        try:
            return self.language_tool.check(text)
        except:
            # Return an empty list if checking fails
            return []

    def evaluate(self, texts, return_mean=True, N=1):
        """
        Evaluate a list of texts and return either the average grammar issues
        or a list of grammar issue counts. Only evaluates every Nth text.

        :param texts: A list of texts to evaluate
        :param return_mean: If True, returns the mean of the issues; 
                            otherwise returns a numpy array of issue counts.
        :param N: Only evaluate grammar issues on every Nth text 
                  (e.g., if N=2, evaluate indices 0, 2, 4, ...).
        """
        # If texts is None or empty, return 0 or empty array
        if not texts:
            return 0 if return_mean else np.array([])
        
        try:
            scores = []
            for i, t in enumerate(texts):
                if i % N == 0:  # only evaluate every Nth index
                    issues = self.find_grammar_issues(t)
                    scores.append(len(issues))
            
            scores = np.array(scores)
            if scores.size == 0:
                # Edge case: if N > len(texts), no scores are collected
                return 0

            return scores.mean() if return_mean else scores
        
        except:
            return 0

    def evaluate_dataframe(self, df, text_column, new_column, early=False, N=1):
        """
        Evaluate a pandas DataFrame, adding a new column with grammar issue counts
        only on every Nth row. Non-Nth rows get NaN.

        :param df: pandas DataFrame containing the text data.
        :param text_column: the name of the column containing the text to evaluate.
        :param new_column: the name of the new column to store the results.
        :param early: if True, only evaluate the first 50 words of each text.
        :param N: Only evaluate rows where row index % N == 0. 
        :return: DataFrame with new column containing grammar issue counts 
                 on every Nth row (others set to NaN).
        """
        # We use df.apply across axis=1 so that we can check the row index
        df[new_column] = df.progress_apply(
            lambda row: len(self.find_grammar_issues(row[text_column], early=early))
            if row["step_num"] % N == 0 or row["step_num"] == -1
            else np.nan,
            axis=1
        )
        
        return df


if __name__ == '__main__':

    # python -m extractors.grammaticality
    
    texts_0 = [
        "I love you",
        "I hate she door not me.",
        "The boy laughed",
        "The boy cried",
        None
    ]

    texts_a = [
        "I know you wanted me to stay",
        "But I can't ignore the crazy visions of me in LA",
        "And I heard that there's a special place",
        "Where boys and girls can all be queens every single day",
    ]

    texts_b = [
        "I'm up and jaws are on the floor",
        "Lovers in the bathroom and a line outside the door",
        "Black lights and a mirrored disco ball",
        "Every night's another reason why I left it all",
    ]

    g_metric = GrammarMetric()

    f_scores = g_metric.evaluate(texts_0, return_mean=False)
    print(f"texts: {texts_0}")
    print(f"fluency_scores (raw): {f_scores}")
    print(f"fluency_scores (mean): {f_scores.mean()}")

    f_scores = g_metric.evaluate(texts_a, return_mean=False)
    print(f"texts: {texts_a}")
    print(f"fluency_scores (raw): {f_scores}")
    print(f"fluency_scores (mean): {f_scores.mean()}")

    f_scores = g_metric.evaluate(texts_b, return_mean=False)
    print(f"texts: {texts_b}")
    print(f"fluency_scores (raw): {f_scores}")
    print(f"fluency_scores (mean): {f_scores.mean()}")