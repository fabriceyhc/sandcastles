
from textdiversity import (
    TokenSemantics, DocumentSemantics, AMR, # semantics
    DependencyParse, ConstituencyParse,     # syntactical
    PartOfSpeechSequence,                   # morphological
    Rhythmic                                # phonological
)
from lexical_diversity import lex_div as ld
from nltk import ngrams
from nltk.tokenize import word_tokenize
import numpy as np

class BaseDiversityMetric:
    def __init__(self, metric):
        self.metric = metric

    def evaluate(self, df):
        return self.metric(df['text'])
    
    def evaluate_before_and_after(self, before_df, after_df, annotate_after_df=True):
        """
        Compares diversity metrics before and after changes.
        Anything lower than 1 means that the changes reduced diversity.
        """
        before_div = self.evaluate(before_df)
        after_div = self.evaluate(after_df)
        div = np.nan_to_num(after_div / before_div)
        return div


class DocumentSemanticDiversity(BaseDiversityMetric):
    def __init__(self):
        super().__init__(DocumentSemantics({"normalize": False}))


class DocumentDependencyParseDiversity(BaseDiversityMetric):
    def __init__(self):
        super().__init__(DependencyParse({"normalize": False}))


class DocumentPartOfSpeechSequenceDiversity(BaseDiversityMetric):
    def __init__(self):
        super().__init__(PartOfSpeechSequence({"normalize": False}))


class MATTRDiversity(BaseDiversityMetric):
    def __init__(self):
        super().__init__(LDHelper().mattr)


class UniqueBigramsDiversity(BaseDiversityMetric):
    def __init__(self):
        super().__init__(UniqueNgramHelper().bigrams)


class LDHelper:

    def _flemmatize(self, corpus):
        flemmas = []
        for doc in corpus:
            flemmas.extend(ld.flemmatize(doc))
        return flemmas

    def ttr(self, coprus):
        return ld.ttr(self._flemmatize(coprus))

    def root_ttr(self, coprus):
        return ld.root_ttr(self._flemmatize(coprus))

    def log_ttr(self, coprus):
        return ld.log_ttr(self._flemmatize(coprus))

    def maas_ttr(self, coprus):
        return ld.maas_ttr(self._flemmatize(coprus))

    def msttr(self, coprus):
        return ld.msttr(self._flemmatize(coprus))

    def mattr(self, coprus):
        return ld.mattr(self._flemmatize(coprus))

    def hdd(self, coprus):
        return ld.hdd(self._flemmatize(coprus))

    def mtld(self, coprus):
        return ld.mtld(self._flemmatize(coprus))

    def mtld_ma_wrap(self, coprus):
        return ld.mtld_ma_wrap(self._flemmatize(coprus))

    def mtld_ma_bid(self, coprus):
        return ld.mtld_ma_bid(self._flemmatize(coprus))


class UniqueNgramHelper:

    def _tokenize(self, corpus):
        tokens = []
        for doc in corpus:
            tokens.extend(word_tokenize(doc))
        return tokens

    def _make_unique(self, n_gram_generator):
        return len(set(list(n_gram_generator)))

    def unigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 1)
        return self._make_unique(n_gram_generator)

    def bigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 2)
        return self._make_unique(n_gram_generator)

    def trigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 3)
        return self._make_unique(n_gram_generator)
    
if __name__ == "__main__":
    
    from datasets import load_dataset
    import numpy as np

    dataset_config = ("glue", "sst2")
    task_name = "sentiment"

    dataset_slice1 = load_dataset(*dataset_config, split="train[0:3]").rename_column("sentence", "text")
    dataset_slice2 = load_dataset(*dataset_config, split="train[3:6]").rename_column("sentence", "text")

    metrics = [
        # DocumentSemanticDiversity(),
        # DocumentDependencyParseDiversity(),
        # DocumentPartOfSpeechSequenceDiversity(),
        MATTRDiversity(),
        UniqueBigramsDiversity()
    ]

    print(f"dataset_slice1: {dataset_slice1['text']}")
    print(f"dataset_slice2: {dataset_slice2['text']}")

    for metric in metrics:
        metric_name = metric.__class__.__name__
        print(f"Calculating {metric_name}...")

        dataset_slice1, div1 = metric.evaluate(dataset_slice1)
        dataset_slice2, div2 = metric.evaluate(dataset_slice2)
        dataset_slice2, div1vs2 = metric.evaluate_before_and_after(dataset_slice1, dataset_slice2)

        print(f"div1_{metric_name}_score: {div1}")
        print(f"div2_{metric_name}_score: {div2}")
        print(f"div1vs2_{metric_name}_scores: {div1vs2}")