import torch
import evaluate
import numpy as np
import transformers

import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import evaluate
from evaluate import logging


_CITATION = """\
"""

_DESCRIPTION = """
Perplexity (PPL) is one of the most common metrics for evaluating language models.
It is defined as the exponentiated average negative log-likelihood of a sequence, calculated with exponent base `e`.
For more information, see https://huggingface.co/docs/transformers/perplexity
"""

_KWARGS_DESCRIPTION = """
Args:
    model_id (str): model used for calculating Perplexity
            NOTE: Perplexity can only be calculated for causal language models.
                    This includes models such as gpt2, causal variations of bert,
                    causal versions of t5, and more (the full list can be found
                    in the AutoModelForCausalLM documentation here:
                    https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModelForCausalLM )
    predictions (list of str): input text, each separate text snippet
        is one list entry.
    batch_size (int): the batch size to run texts through the model. Defaults to 16.
    add_start_token (bool): whether to add the start token to the texts,
        so the perplexity can include the probability of the first word. Defaults to True.
    device (str): device to run on, defaults to 'cuda' when available
Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.
Examples:
    Example 1:
        >>> perplexity = evaluate.load("perplexity", module_type="metric")
        >>> input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]
        >>> results = perplexity.compute(model_id='gpt2',
        ...                              add_start_token=False,
        ...                              predictions=input_texts) # doctest:+ELLIPSIS
        >>> print(list(results.keys()))
        ['perplexities', 'mean_perplexity']
        >>> print(round(results["mean_perplexity"], 0))
        647.0
        >>> print(round(results["perplexities"][0], 0))
        32.0
    Example 2:
        >>> from datasets import load_dataset
        >>> perplexity = evaluate.load("perplexity", module_type="metric")
        >>> input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:10] # doctest: +SKIP
        >>> input_texts = [s for s in input_texts if s!='']
        >>> results = perplexity.compute(model_id='gpt2',
        ...                              predictions=input_texts)
        >>> print(list(results.keys()))
        ['perplexities', 'mean_perplexity']
        >>> print(round(results["mean_perplexity"], 2)) # doctest: +SKIP
        576.76
        >>> print(round(results["perplexities"][0], 2)) # doctest: +SKIP
        889.28
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Perplexity(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    def _compute(
        self, predictions, model_id, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None
    ):

        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model.resize_token_embeddings(len(tokenizer))

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


class FluencyMetric:
    def __init__(self, model_id='gpt2', batch_size=1, device="cuda") -> None:
        """
        Use gpt2 to measure how perplexing / surprising a given text is 
        to a well trained language model. When used on text that we know
        is natural / human sounding, then perplexity is a measure of 
        model quality. However, when we trust that the language model is
        pretty good already and we aren't sure about the quality of the 
        text, we can use perplexity to measure text naturalness. 
        :Package Requirements:
            * pip install evaluate
        :Language: english
        """
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self.metric = Perplexity()  # Or load("perplexity") if using evaluate.load

    def evaluate(self, texts, return_mean=True, N=1):
        """
        Evaluate perplexity on a list of texts. If N > 1, only evaluate
        on texts whose index (0-based) is divisible by N.
        
        :param texts: list of text strings.
        :param return_mean: if True, returns the mean perplexity; 
                            otherwise returns the array of perplexities.
        :param N: skip texts whose index is not divisible by N. Defaults to 1 (evaluate all).
        :return: either the mean of perplexities or an array of perplexities (for the subset).
        """
        if not texts:
            return 0 if return_mean else np.array([])

        # Select only texts where i % N == 0
        selected_texts = [txt for i, txt in enumerate(texts) if i % N == 0]
        if len(selected_texts) == 0:
            # Edge case: if N > len(texts), no texts are selected
            return 0 if return_mean else np.array([])

        # Compute perplexities on the selected subset
        results = self.metric._compute(
            predictions=selected_texts,
            model_id=self.model_id,
            max_length=1024,
            device=self.device,
            batch_size=self.batch_size
        )['perplexities']

        scores = np.array(results)
        return scores.mean() if return_mean else scores

    def evaluate_single_text(self, text):
        """
        Compute the perplexity for a single text. Returns np.nan if text is not valid.
        """
        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            return np.nan

        try:
            results = self.metric._compute(
                predictions=[text],
                model_id=self.model_id,
                max_length=1024,
                device=self.device,
                batch_size=1  # Single text
            )
            # 'perplexities' is a list, so return the first (only) item
            return results['perplexities'][0]
        except Exception:
            return np.nan

    def evaluate_dataframe(self, df, text_column, new_column, N=1):
        """
        Evaluate a pandas DataFrame row-by-row, adding a new column with perplexity scores.
        We only compute the perplexity for rows whose index is divisible by N; 
        other rows get NaN.

        :param df: pandas DataFrame containing the text data.
        :param text_column: name of the column containing the text to evaluate.
        :param new_column: name of the new column to store results.
        :param N: only evaluate rows where row.index % N == 0.
        :return: DataFrame with new column containing perplexity scores 
                 (NaN for rows that are not multiples of N).
        """
        df[new_column] = df.progress_apply(
            lambda row: self.evaluate_single_text(row[text_column])
            if row["step_num"] % N == 0 or row["step_num"] == -1
            else np.nan, 
            axis=1
        )
        return df

if __name__ == '__main__':
    
    texts_0 = [
        "I love you.",
        "I hate she door not me.",
        "The boy laughed.",
        "The boy cried.",
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

    f_metric = FluencyMetric()

    f_scores = f_metric.evaluate(texts_0, return_mean=False)
    print(f"texts: {texts_0}")
    print(f"fluency_scores (raw): {f_scores}")
    print(f"fluency_scores (mean): {f_scores.mean()}")

    f_scores = f_metric.evaluate(texts_a, return_mean=False)
    print(f"texts: {texts_a}")
    print(f"fluency_scores (raw): {f_scores}")
    print(f"fluency_scores (mean): {f_scores.mean()}")

    f_scores = f_metric.evaluate(texts_b, return_mean=False)
    print(f"texts: {texts_b}")
    print(f"fluency_scores (raw): {f_scores}")
    print(f"fluency_scores (mean): {f_scores.mean()}")