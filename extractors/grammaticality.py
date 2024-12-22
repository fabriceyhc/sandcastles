import language_tool_python
import numpy as np

class GrammarMetric:
    def __init__(self) -> None:
        """
        Use language_tool_python to check grammer.
        :Package Requirements:
            * pip install language_tool_python
        :Language: english
        """
        self.language_tool = language_tool_python.LanguageTool('en-US')

    def find_grammar_issues(self, text, early=False):
        if early:
            text = ' '.join(text.split()[:50])
        return self.language_tool.check(text)
    

    def evaluate(self, texts, return_mean=True):
        scores = [len(self.find_grammar_issues(t)) for t in texts]
        scores = np.array(scores)
        return scores.mean() if return_mean else scores

    def evaluate_dataframe(self, df, text_column, new_column, early=False):
        """
        Evaluate a pandas DataFrame, adding a new column with grammar issue counts.
        
        :param df: pandas DataFrame containing the text data.
        :param text_column: the name of the column containing the text to evaluate.
        :param new_column: the name of the new column to store the results.
        :return: DataFrame with new column containing grammar issue counts.
        """
        df[new_column] = df[text_column].apply(lambda text: len(self.find_grammar_issues(text, early=early)))
        return df
    

if __name__ == '__main__':
    
    texts_0 = [
        "I love you",
        "I hate she door not me.",
        "The boy laughed",
        "The boy cried",
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