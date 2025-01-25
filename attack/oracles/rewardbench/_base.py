import numpy as np

class BaseRewardBenchOracle:
    """
    A reusable base class that provides batch-evaluation methods. 
    Child oracles only need to define the '_score_example' method.
    """

    def score_samples(self, prompts, texts, return_mean=True):
        """
        Evaluate pairs of (prompt, text) by calling '_score_example' for each pair.

        :param prompts: List of prompt strings.
        :param texts: List of text/response strings.
        :param return_mean: If True, returns the mean of scores. Otherwise returns a NumPy array of scores.
        :return: float (mean) or np.array (all scores).
        """
        all_scores = []
        for prompt, text in zip(prompts, texts):
            score = self._score_example(prompt, text)
            all_scores.append(score)
        all_scores = np.array(all_scores)
        return all_scores.mean() if return_mean else all_scores

    def score_dataframe(self, df, prompt_column, text_column, new_column):
        """
        Evaluate a pandas DataFrame, adding a new column with quality scores.

        :param df: pandas DataFrame containing prompts and texts.
        :param prompt_column: Name of the column with prompt strings.
        :param text_column: Name of the column with text/response strings.
        :param new_column: Name of the new column to store computed quality scores.
        :return: DataFrame with new column containing quality scores.
        """
        prompts = df[prompt_column].tolist()
        texts = df[text_column].tolist()
        scores = self.score_samples(prompts, texts, return_mean=False)
        df[new_column] = scores
        return df

    def _score_example(self, prompt, text):
        """
        Abstract method that child classes must implement to score a single (prompt, text) pair.
        
        :param prompt: The user query or prompt.
        :param text: The candidate response to the prompt.
        :return: A single float score.
        """
        raise NotImplementedError("Child classes must implement '_score_example'.")
