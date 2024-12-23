from difflib import SequenceMatcher
import re
import numpy as np

class EditsMetric:
    def __init__(self) -> None:
        """
        Use difflib to see how many words have changed.
        """

    def diff_analysis(self, response1, response2):
        text1 = re.split(r'(\S+)', response1)
        line1 = text1[1::2]
        text2 = re.split(r'(\S+)', response2)
        line2 = text2[1::2]

        different_word_count = 0
        matcher = SequenceMatcher(None, line1, line2)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':
                different_word_count += (i2 - i1)  # Words in response1 but not in response2
            elif tag == 'insert':
                different_word_count += (j2 - j1)  # Words in response2 but not in response1
            elif tag == 'replace':
                different_word_count += max(i2 - i1, j2 - j1)  # Replaced words

        return different_word_count

    def evaluate(self, texts1, texts2, return_mean=True):
        scores = [self.diff_analysis(t1, t2) for t1, t2 in zip(texts1, texts2)]
        scores = np.array(scores)
        return scores.mean() if return_mean else scores

    def evaluate_dataframe(self, df, current_text_column, mutated_text_column, new_column):
        """
        Evaluate a pandas DataFrame, adding a new column with edit counts.
        
        :param df: pandas DataFrame containing the text data.
        :param current_text_column: the name of the column containing the original text.
        :param mutated_text_column: the name of the column containing the modified text (can have null values).
        :param new_column: the name of the new column to store the results.
        :return: DataFrame with new column containing edit counts.
        """
        
        # Apply the diff_analysis for each row in the DataFrame
        df[new_column] = df.apply(lambda row: self.diff_analysis(row[current_text_column], row[mutated_text_column]), axis=1)
        return df


if __name__ == '__main__':

    # RUN: python -m extractors.edits
    
    texts1 = [
        "I love you",
        "I hate she door not me.",
        "The boy laughed",
        "The boy cried",
    ]

    texts2 = [
        "I love you so much",
        "I hate she door me.",
        "The boy choopa",
        "A girl shrieked",
    ]

    e_metric = EditsMetric()

    e_scores = e_metric.evaluate(texts1, texts2, return_mean=False)
    print(f"texts: {texts1, texts2}")
    print(f"edit_count (raw): {e_scores}")
    print(f"edit_count (mean): {e_scores.mean()}")