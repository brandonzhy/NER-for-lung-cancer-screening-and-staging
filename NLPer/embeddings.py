import os
import re
import logging
from abc import abstractmethod
from collections import Counter
from pathlib import Path
from typing import List, Union, Dict

import gensim
import numpy as np
import torch
from bpemb import BPEmb
from deprecated import deprecated
from torch.nn import ParameterList, Parameter

from pytorch_transformers import (
    BertTokenizer,
    BertModel,
    GPT2Model,
    GPT2Tokenizer,
    XLNetTokenizer,
    XLNetModel,
    PreTrainedTokenizer,
    PreTrainedModel,
)

import NLPer
from NLPer.data import Corpus
from .data import Dictionary, Token, Sentence
from .file_utils import cached_path, open_inside_zip
from .training_utils import log_line

log = logging.getLogger("NLPer")


class Embeddings(torch.nn.Module):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        pass

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        everything_embedded: bool = True

        if self.embedding_type == "word-level":
            for sentence in sentences:
                for token in sentence.tokens:
                    if self.name not in token._embeddings.keys():
                        everything_embedded = False
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys():
                    everything_embedded = False

        if not everything_embedded or not self.static_embeddings:
            self._add_embeddings_internal(sentences)

        return sentences

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        pass


class TokenEmbeddings(Embeddings):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) -> str:
        return "word-level"


class StackedEmbeddings(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[TokenEmbeddings]):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            embedding.name = f"{str(i)}-{embedding.name}"
            self.add_module(f"list_embedding_{str(i)}", embedding)

        self.name: str = "Stack"
        self.static_embeddings: bool = True

        self.__embedding_type: str = embeddings[0].embedding_type

        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(
        self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True
    ):
        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences

    def __str__(self):
        return f'StackedEmbeddings [{",".join([str(e) for e in self.embeddings])}]'

    def get_names(self) -> List[str]:
        """Returns a list of embedding names. In most cases, it is just a list with one item, namely the name of
        this embedding. But in some cases, the embedding is made up by different embeddings (StackedEmbedding).
        Then, the list contains the names of all embeddings in the stack."""
        names = []
        for embedding in self.embeddings:
            names.extend(embedding.get_names())
        return names

    def get_named_embeddings_dict(self) -> Dict:

        named_embeddings_dict = {}
        for embedding in self.embeddings:
            named_embeddings_dict.update(embedding.get_named_embeddings_dict())

        return named_embeddings_dict


class FastTextEmbeddings(TokenEmbeddings):
    """FastText Embeddings with oov functionality"""

    def __init__(self, embeddings: str, use_local: bool = True, field: str = None):
        """
        Initializes fasttext word embeddings. Constructor downloads required embedding file and stores in cache
        if use_local is False.

        :param embeddings: path to your embeddings '.bin' file
        :param use_local: set this to False if you are using embeddings from a remote source
        """

        cache_dir = Path("embeddings")

        if use_local:
            if not Path(embeddings).exists():
                raise ValueError(
                    f'The given embeddings "{embeddings}" is not available or is not a valid path.'
                )
        else:
            embeddings = cached_path(f"{embeddings}", cache_dir=cache_dir)

        self.embeddings = embeddings

        self.name: str = str(embeddings)

        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(
            str(embeddings)
        )

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size

        self.field = field
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                try:
                    word_embedding = self.precomputed_word_embeddings[word]
                except:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return f"'{self.embeddings}'"

    """Standard static word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code or custom
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """
        self.embeddings = embeddings

        old_base_path = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/"
        )
        base_path = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/"
        )
        embeddings_path_v4 = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/"
        )
        embeddings_path_v4_1 = "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4.1/"

        cache_dir = Path("embeddings")

        # GLOVE embeddings
        if embeddings.lower() == "glove" or embeddings.lower() == "en-glove":
            cached_path(f"{old_base_path}glove.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(
                f"{old_base_path}glove.gensim", cache_dir=cache_dir
            )

        # TURIAN embeddings
        elif embeddings.lower() == "turian" or embeddings.lower() == "en-turian":
            cached_path(
                f"{embeddings_path_v4_1}turian.vectors.npy", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{embeddings_path_v4_1}turian", cache_dir=cache_dir
            )

        # KOMNINOS embeddings
        elif embeddings.lower() == "extvec" or embeddings.lower() == "en-extvec":
            cached_path(
                f"{old_base_path}extvec.gensim.vectors.npy", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{old_base_path}extvec.gensim", cache_dir=cache_dir
            )

        # FT-CRAWL embeddings
        elif embeddings.lower() == "crawl" or embeddings.lower() == "en-crawl":
            cached_path(
                f"{base_path}en-fasttext-crawl-300d-1M.vectors.npy", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{base_path}en-fasttext-crawl-300d-1M", cache_dir=cache_dir
            )

        # FT-CRAWL embeddings
        elif (
            embeddings.lower() == "news"
            or embeddings.lower() == "en-news"
            or embeddings.lower() == "en"
        ):
            cached_path(
                f"{base_path}en-fasttext-news-300d-1M.vectors.npy", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{base_path}en-fasttext-news-300d-1M", cache_dir=cache_dir
            )

        # twitter embeddings
        elif embeddings.lower() == "twitter" or embeddings.lower() == "en-twitter":
            cached_path(
                f"{old_base_path}twitter.gensim.vectors.npy", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{old_base_path}twitter.gensim", cache_dir=cache_dir
            )

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 2:
            cached_path(
                f"{embeddings_path_v4}{embeddings}-wiki-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings = cached_path(
                f"{embeddings_path_v4}{embeddings}-wiki-fasttext-300d-1M",
                cache_dir=cache_dir,
            )

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 7 and embeddings.endswith("-wiki"):
            cached_path(
                f"{embeddings_path_v4}{embeddings[:2]}-wiki-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings = cached_path(
                f"{embeddings_path_v4}{embeddings[:2]}-wiki-fasttext-300d-1M",
                cache_dir=cache_dir,
            )

        # two-letter language code crawl embeddings
        elif len(embeddings.lower()) == 8 and embeddings.endswith("-crawl"):
            cached_path(
                f"{embeddings_path_v4}{embeddings[:2]}-crawl-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings = cached_path(
                f"{embeddings_path_v4}{embeddings[:2]}-crawl-fasttext-300d-1M",
                cache_dir=cache_dir,
            )

        elif not Path(embeddings).exists():
            raise ValueError(
                f'The given embeddings "{embeddings}" is not available or is not a valid path.'
            )

        self.name: str = str(embeddings)
        self.static_embeddings = True

        if str(embeddings).endswith(".bin"):
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                str(embeddings), binary=True
            )
        else:
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(
                str(embeddings)
            )

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif (
                    re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "#", word.lower())
                    ]
                elif (
                    re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings
                ):
                    word_embedding = self.precomputed_word_embeddings[
                        re.sub(r"\d", "0", word.lower())
                    ]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        # fix serialized models
        if "embeddings" not in self.__dict__:
            self.embeddings = self.name

        return f"'{self.embeddings}'"


class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors.
    This method was proposed by Liu et al. (2019) in the paper:
    "Linguistic Knowledge and Transferability of Contextual Representations" (https://arxiv.org/abs/1903.08855)

    The implementation is copied and slightly modified from the allennlp repository and is licensed under Apache 2.0.
    It can be found under:
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py.
    """

    def __init__(self, mixture_size: int) -> None:
        """
        Inits scalar mix implementation.
        ``mixture = gamma * sum(s_k * tensor_k)`` where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.
        :param mixture_size: size of mixtures (usually the number of layers)
        """
        super(ScalarMix, self).__init__()
        self.mixture_size = mixture_size

        initial_scalar_parameters = [0.0] * mixture_size

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([initial_scalar_parameters[i]]).to(NLPer.device),
                    requires_grad=False,
                )
                for i in range(mixture_size)
            ]
        )
        self.gamma = Parameter(
            torch.FloatTensor([1.0]).to(NLPer.device), requires_grad=False
        )

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        :param tensors: list of input tensors
        :return: computed weighted average of input tensors
        """
        if len(tensors) != self.mixture_size:
            log.error(
                "{} tensors were passed, but the module was initialized to mix {} tensors.".format(
                    len(tensors), self.mixture_size
                )
            )

        normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]), dim=0
        )
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)

def _extract_embeddings(
    hidden_states: List[torch.FloatTensor],
    layers: List[int],
    pooling_operation: str,
    subword_start_idx: int,
    subword_end_idx: int,
    use_scalar_mix: bool = False,
) -> List[torch.FloatTensor]:
    """
    Extracts subword embeddings from specified layers from hidden states.
    :param hidden_states: list of hidden states from model
    :param layers: list of layers
    :param pooling_operation: pooling operation for subword embeddings (supported: first, last, first_last and mean)
    :param subword_start_idx: defines start index for subword
    :param subword_end_idx: defines end index for subword
    :param use_scalar_mix: determines, if scalar mix should be used
    :return: list of extracted subword embeddings
    """
    subtoken_embeddings: List[torch.FloatTensor] = []

    for layer in layers:
        # if len(layer) == 0:
        # current_embeddings = hidden_states[layer][0][subword_start_idx:subword_end_idx]
        current_embeddings = hidden_states[layer][0]

        first_embedding: torch.FloatTensor = current_embeddings[0] if len(current_embeddings)>0 else current_embeddings
        if pooling_operation == "first_last":
            last_embedding: torch.FloatTensor = current_embeddings[-1]
            final_embedding: torch.FloatTensor = torch.cat(
                [first_embedding, last_embedding]
            )
        elif pooling_operation == "last":
            final_embedding: torch.FloatTensor = current_embeddings[-1]
        elif pooling_operation == "mean":
            all_embeddings: List[torch.FloatTensor] = [
                embedding.unsqueeze(0) for embedding in current_embeddings
            ]
            final_embedding: torch.FloatTensor = torch.mean(
                torch.cat(all_embeddings, dim=0), dim=0
            )
        else:
            final_embedding: torch.FloatTensor = first_embedding

        subtoken_embeddings.append(final_embedding)

    if use_scalar_mix:
        sm = ScalarMix(mixture_size=len(subtoken_embeddings))
        sm_embeddings = sm(subtoken_embeddings)

        subtoken_embeddings = [sm_embeddings]

    return subtoken_embeddings


def _build_token_subwords_mapping(
    sentence: Sentence, tokenizer: PreTrainedTokenizer
) -> Dict[int, int]:
    """ Builds a dictionary that stores the following information:
    Token index (key) and number of corresponding subwords (value) for a sentence.

    :param sentence: input sentence
    :param tokenizer: PyTorch-Transformers tokenization object
    :return: dictionary of token index to corresponding number of subwords
    """
    token_subwords_mapping: Dict[int, int] = {}

    for token in sentence.tokens:
        token_text = token.text

        subwords = tokenizer.tokenize(token_text)

        token_subwords_mapping[token.idx] = len(subwords)

    return token_subwords_mapping


def _build_token_subwords_mapping_roberta(sentence: Sentence, model) -> Dict[int, int]:
    """ Builds a dictionary that stores the following information:
    Token index (key) and number of corresponding subwords (value) for a sentence.

    :param sentence: input sentence
    :param model: RoBERTa model
    :return: dictionary of token index to corresponding number of subwords
    """
    token_subwords_mapping: Dict[int, int] = {}

    for token in sentence.tokens:
        token_text = token.text

        # Leading spaces are needed for GPT2 BPE tokenization in RoBERTa (except at BOS):
        # ``roberta.encode(' world').tolist()`` -> ``[0, 232, 2]``
        # ``roberta.encode('world').tolist()``  -> ``[0, 8331, 2]``
        padding = "" if token.idx == 1 else " "

        current_subwords = model.encode(padding + token_text)

        # ``roberta.encode(' world').tolist()`` will result in ``[0, 232, 2]``:
        # 0 and 2 are special symbols (`<s>` and `</s>`), so ignore them in subword length calculation
        token_subwords_mapping[token.idx] = len(current_subwords) - 2

    return token_subwords_mapping


def _build_token_subwords_mapping_gpt2(
    sentence: Sentence, tokenizer: PreTrainedTokenizer
) -> Dict[int, int]:
    """ Builds a dictionary that stores the following information:
    Token index (key) and number of corresponding subwords (value) for a sentence.

    :param sentence: input sentence
    :param tokenizer: PyTorch-Transformers tokenization object
    :return: dictionary of token index to corresponding number of subwords
    """
    token_subwords_mapping: Dict[int, int] = {}

    for token in sentence.tokens:
        # Dummy token is needed to get the actually token tokenized correctly with special ``Ġ`` symbol

        if token.idx == 1:
            token_text = token.text
            subwords = tokenizer.tokenize(token_text)
        else:
            token_text = "X " + token.text
            subwords = tokenizer.tokenize(token_text)[1:]

        token_subwords_mapping[token.idx] = len(subwords)

    return token_subwords_mapping


def _get_transformer_sentence_embeddings(
    sentences: List[Sentence],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    name: str,
    layers: List[int],
    pooling_operation: str,
    use_scalar_mix: bool,
    bos_token: str = None,
    eos_token: str = None,
) -> List[Sentence]:
    """
    Builds sentence embeddings for Transformer-based architectures.
    :param sentences: input sentences
    :param tokenizer: tokenization object
    :param model: model object
    :param name: name of the Transformer-based model
    :param layers: list of layers
    :param pooling_operation: defines pooling operation for subword extraction
    :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)
    :param bos_token: defines begin of sentence token (used for left padding)
    :param eos_token: defines end of sentence token (used for right padding)
    :return: list of sentences (each token of a sentence is now embedded)
    """
    with torch.no_grad():
        for sentence in sentences:
            token_subwords_mapping: Dict[int, int] = {}

            if name.startswith("roberta"):
                token_subwords_mapping = _build_token_subwords_mapping_roberta(
                    sentence=sentence, model=model
                )
            elif name.startswith("gpt2"):
                token_subwords_mapping = _build_token_subwords_mapping_gpt2(
                    sentence=sentence, tokenizer=tokenizer
                )
            else:
                token_subwords_mapping = _build_token_subwords_mapping(
                    sentence=sentence, tokenizer=tokenizer
                )

            if name.startswith("roberta"):
                subwords = model.encode(sentence.to_tokenized_string())
            else:
                subwords = tokenizer.tokenize(sentence.to_tokenized_string())

            offset = 0

            if bos_token:
                subwords = [bos_token] + subwords
                offset = 1

            if eos_token:
                subwords = subwords + [eos_token]

            if not name.startswith("roberta"):
                indexed_tokens = tokenizer.convert_tokens_to_ids(subwords)
                tokens_tensor = torch.tensor([indexed_tokens])
                tokens_tensor = tokens_tensor.to(NLPer.device)

                hidden_states = model(tokens_tensor)[-1]
            else:
                hidden_states = model.extract_features(
                    subwords, return_all_hiddens=True
                )
                offset = 1

            for token in sentence.tokens:
                len_subwords = token_subwords_mapping[token.idx]

                subtoken_embeddings = _extract_embeddings(
                    hidden_states=hidden_states,
                    layers=layers,
                    pooling_operation=pooling_operation,
                    subword_start_idx=offset,
                    subword_end_idx=offset + len_subwords,
                    use_scalar_mix=use_scalar_mix,
                )

                offset += len_subwords

                final_subtoken_embedding = torch.cat(subtoken_embeddings)
                token.set_embedding(name, final_subtoken_embedding)

    return sentences


class XLNetEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "xlnet-large-cased",
        layers: str = "1",
        pooling_operation: str = "first_last",
        use_scalar_mix: bool = False,
    ):
        """XLNet embeddings, as proposed in Yang et al., 2019.
        :param pretrained_model_name_or_path: name or path of XLNet model
        :param layers: comma-separated list of layers
        :param pooling_operation: defines pooling operation for subwords
        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)
        """
        super().__init__()

        self.tokenizer = XLNetTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = XLNetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, output_hidden_states=True
        )
        self.name = pretrained_model_name_or_path
        self.layers: List[int] = [int(layer) for layer in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True

        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        self.model.to(NLPer.device)
        self.model.eval()

        sentences = _get_transformer_sentence_embeddings(
            sentences=sentences,
            tokenizer=self.tokenizer,
            model=self.model,
            name=self.name,
            layers=self.layers,
            pooling_operation=self.pooling_operation,
            use_scalar_mix=self.use_scalar_mix,
            bos_token="<s>",
            eos_token="</s>",
        )

        return sentences

    def extra_repr(self):
        return "model={}".format(self.name)

    def __str__(self):
        return self.name


class GPT2Embeddings(TokenEmbeddings):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "gpt2-medium",
        layers: str = "1",
        pooling_operation: str = "first_last",
        use_scalar_mix: bool = False,
    ):
        """OpenAI GPT-2 embeddings, as proposed in Radford et al. 2019.
        :param pretrained_model_name_or_path: name or path of OpenAI GPT-2 model
        :param layers: comma-separated list of layers
        :param pooling_operation: defines pooling operation for subwords
        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)
        """
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = GPT2Model.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, output_hidden_states=True
        )
        self.name = pretrained_model_name_or_path
        self.layers: List[int] = [int(layer) for layer in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True

        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        self.model.to(NLPer.device)
        self.model.eval()

        sentences = _get_transformer_sentence_embeddings(
            sentences=sentences,
            tokenizer=self.tokenizer,
            model=self.model,
            name=self.name,
            layers=self.layers,
            pooling_operation=self.pooling_operation,
            use_scalar_mix=self.use_scalar_mix,
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
        )

        return sentences


class BertEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        bert_model_or_path: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
        use_scalar_mix: bool = False,
        use_albert: bool = False
    ):
        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model_or_path: name of BERT model ('') or directory path containing custom model, configuration file
        and vocab file (names of three files should be - config.json, pytorch_model.bin/model.chkpt, vocab.txt)
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first)
        """
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path)

        if use_albert:
            self.model = AlbertModel.from_pretrained(
                pretrained_model_name_or_path=bert_model_or_path, output_hidden_states=True
            )
        else:
            self.model = BertModel.from_pretrained(
                pretrained_model_name_or_path=bert_model_or_path, output_hidden_states=True
            )
        # self.model = BertModel.from_pretrained(
        #     pretrained_model_name_or_path=bert_model_or_path, output_hidden_states=True
        # )
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.name = str(bert_model_or_path)
        self.static_embeddings = True

    class BertInputFeatures(object):
        """Private helper class for holding BERT-formatted features"""

        def __init__(
            self,
            unique_id,
            tokens,
            input_ids,
            input_mask,
            input_type_ids,
            token_subtoken_count,
        ):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids
            self.token_subtoken_count = token_subtoken_count

    def _convert_sentences_to_features(
        self, sentences, max_sequence_length: int
    ) -> [BertInputFeatures]:

        max_sequence_length = max_sequence_length + 2

        features: List[BertEmbeddings.BertInputFeatures] = []
        for (sentence_index, sentence) in enumerate(sentences):

            bert_tokenization: List[str] = []
            token_subtoken_count: Dict[int, int] = {}

            # 中文没有分割符，故token就是这句话
            for token in sentence:
                # 需要用token的分词工具来将句子分开（必须传入的是text，tokenize实际上调用的是str的split（）方法，只有文本才有split（）方法）
                subtokens = self.tokenizer.tokenize(token.text)
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)

            if len(bert_tokenization) > max_sequence_length - 2:
                bert_tokenization = bert_tokenization[0 : (max_sequence_length - 2)]

            tokens = []
            # input_type_ids = []
            tokens.append("[CLS]")
            # input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                # input_type_ids.append(0)
            tokens.append("[SEP]")
            # input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            # print('length of input_ids is :',len(input_ids))
            # print('length of tokens is :', len(tokens))

            input_mask = [1] * len(input_ids)
            input_type_ids = [0]*len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_sequence_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            features.append(
                BertEmbeddings.BertInputFeatures(
                    unique_id=sentence_index,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids,
                    token_subtoken_count=token_subtoken_count,
                )
            )

        return features

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""

        # first, find longest sentence in batch
        longest_sentence_in_batch: int = len(
            max(
                [
                    self.tokenizer.tokenize(sentence.to_tokenized_string())
                    for sentence in sentences
                ],
                key=len,
            )
        )

        # prepare id maps for BERT model
        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch
        )
        all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(
            NLPer.device
        )
        all_input_masks = torch.LongTensor([f.input_mask for f in features]).to(
            NLPer.device
        )

        # put encoded batch through BERT model to get all hidden states of all encoder layers
        self.model.to(NLPer.device)
        self.model.eval()
        _, _, all_encoder_layers = self.model(
            all_input_ids, token_type_ids=None, attention_mask=all_input_masks
        )

        with torch.no_grad():

            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                # get aggregated embeddings for each BERT-subtoken in sentence
                subtoken_embeddings = []
                # 这里token_index是由tokens来的，而tokens是分词后的结果
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        if self.use_scalar_mix:
                            layer_output = all_encoder_layers[int(layer_index)][
                                sentence_index
                            ]
                        else:
                            layer_output = (
                                all_encoder_layers[int(layer_index)]
                                .detach()
                                .cpu()[sentence_index]
                            )
                        all_layers.append(layer_output[token_index])

                    if self.use_scalar_mix:
                        sm = ScalarMix(mixture_size=len(all_layers))
                        sm_embeddings = sm(all_layers)
                        all_layers = [sm_embeddings]

                    subtoken_embeddings.append(torch.cat(all_layers))

                # get the current sentence object
                token_idx = 0
                for token in sentence:
                    # add concatenated embedding to sentence
                    token_idx += 1

                    if self.pooling_operation == "first":
                        # use first subword embedding if pooling operation is 'first'
                        token.set_embedding(self.name, subtoken_embeddings[token_idx])
                    else:
                        # otherwise, do a mean over all subwords in token
                        embeddings = subtoken_embeddings[
                            token_idx : token_idx
                            + feature.token_subtoken_count[token.idx]
                        ]
                        embeddings = [
                            embedding.unsqueeze(0) for embedding in embeddings
                        ]
                        mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        token.set_embedding(self.name, mean)

                    token_idx += feature.token_subtoken_count[token.idx] - 1

        return sentences

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return (
            len(self.layer_indexes) * self.model.config.hidden_size
            if not self.use_scalar_mix
            else self.model.config.hidden_size
        )


def replace_with_language_code(string: str):
    string = string.replace("arabic-", "ar-")
    string = string.replace("basque-", "eu-")
    string = string.replace("bulgarian-", "bg-")
    string = string.replace("croatian-", "hr-")
    string = string.replace("czech-", "cs-")
    string = string.replace("danish-", "da-")
    string = string.replace("dutch-", "nl-")
    string = string.replace("farsi-", "fa-")
    string = string.replace("persian-", "fa-")
    string = string.replace("finnish-", "fi-")
    string = string.replace("french-", "fr-")
    string = string.replace("german-", "de-")
    string = string.replace("hebrew-", "he-")
    string = string.replace("hindi-", "hi-")
    string = string.replace("indonesian-", "id-")
    string = string.replace("italian-", "it-")
    string = string.replace("japanese-", "ja-")
    string = string.replace("norwegian-", "no")
    string = string.replace("polish-", "pl-")
    string = string.replace("portuguese-", "pt-")
    string = string.replace("slovenian-", "sl-")
    string = string.replace("spanish-", "es-")
    string = string.replace("swedish-", "sv-")
    return string
