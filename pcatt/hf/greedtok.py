# coding=utf-8
# Original Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changes were made to fit GreedTok
# The input variables to the various methods were preserved

import copy
import json
import os
import regex
import warnings
from collections import OrderedDict, UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    Callable,
)
import numpy as np
import multiprocessing
from enum import Enum
from functools import partial

try:
    import tensorflow as tf

    assert hasattr(tf, "__version__") and int(tf.__version__[0]) >= 2
    _tf_available = True
except (ImportError, AssertionError):
    _tf_available = False

try:
    import torch

    _torch_available = True
except:
    _torch_available = False

try:
    import jax

    _flax_available = _jax_available = True
except:
    _flax_available = _jax_available = False


@dataclass(frozen=True, eq=True)
class AddedToken:
    """
    AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
    way it should behave.
    """

    content: str = field(default_factory=str)
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = True

    def __getstate__(self):
        return self.__dict__


VERY_LARGE_INTEGER = int(
    1e30
)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(
    1e20
)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]


# Slow tokenizers used to be saved in three separated files
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.txt"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


class CharSpan(NamedTuple):
    """
    Character span in the original string.

    Args:
        start (`int`): Index of the first character in the original string.
        end (`int`): Index of the character following the last character in the original string.
    """

    start: int
    end: int


class TokenSpan(NamedTuple):
    """
    Token span in an encoded string (list of tokens).

    Args:
        start (`int`): Index of the first token in the span.
        end (`int`): Index of the token following the last token in the span.
    """

    start: int
    end: int


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"


class BatchEncoding(UserDict):
    """
    Holds the output of the [`~tokenization_utils_base.PreTrainedTokenizerBase.__call__`],
    [`~tokenization_utils_base.PreTrainedTokenizerBase.encode_plus`] and
    [`~tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus`] methods (tokens, attention_masks, etc).

    This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
    utility methods to map from word/character space to token space.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the `__call__`/`encode_plus`/`batch_encode_plus` methods
            ('input_ids', 'attention_mask', etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
        prepend_batch_axis (`bool`, *optional*, defaults to `False`):
            Whether or not to add a batch axis when converting to tensors (see `tensor_type` above).
        n_sequences (`Optional[int]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        tensor_type: Union[None, str, TensorType] = None,
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    ):
        super().__init__(data)
        self._n_sequences = n_sequences
        self.convert_to_tensors(
            tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis
        )

    @property
    def n_sequences(self) -> Optional[int]:
        """
        `Optional[int]`: The number of sequences used to generate each sample from the batch encoded in this
        [`BatchEncoding`]. Currently can be one of `None` (unknown), `1` (a single sentence) or `2` (a pair of
        sentences)
        """
        return self._n_sequences

    @property
    def is_fast(self) -> bool:
        """
        Default false as rust backend not used.
        """
        return False

    def __getitem__(self, item: Union[int, str]) -> Union[Any, EncodedInput]:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the `tokenizers.Encoding` for batch item with index `key`.

        If the key is a slice, returns the value of the dict associated to `key` ('input_ids', 'attention_mask', etc.)
        with the constraint of slice.
        """
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        elif isinstance(item, slice):
            return {key: self.data[key][slice] for key in self.data.keys()}
        else:
            raise KeyError(
                "Invalid key. Only three types of key are available: "
                "(1) string, (2) integers for backend Encoding, and (3) slices for data subsetting."
            )

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data, "encodings": self._encodings}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

        if "encodings" in state:
            self._encodings = state["encodings"]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    # After this point:
    # Extended properties and methods only available for fast (Rust-based) tokenizers
    # provided by HuggingFace tokenizers library.

    def convert_to_tensors(
        self,
        tensor_type: Optional[Union[str, TensorType]] = None,
        prepend_batch_axis: bool = False,
    ):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.TENSORFLOW:
            try:
                import tensorflow as tf

                as_tensor = tf.constant
                is_tensor = tf.is_tensor
            except ImportError:
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )

        elif tensor_type == TensorType.PYTORCH:
            try:
                import torch

                as_tensor = torch.tensor
                is_tensor = torch.is_tensor
            except ImportError:
                raise ImportError(
                    "Unable to convert output to PyTorch tensors format, PyTorch is not installed."
                )

        elif tensor_type == TensorType.JAX:
            try:
                import jax.numpy as jnp

                as_tensor = jnp.array
                is_tensor = lambda x: isinstance(x, jnp.ndarray)
            except ImportError:
                raise ImportError(
                    "Unable to convert output to JAX tensors format, JAX is not installed."
                )
        else:

            def as_tensor(value, dtype=None):
                if isinstance(value, (list, tuple)) and isinstance(
                    value[0], (list, tuple, np.ndarray)
                ):
                    value_lens = [len(val) for val in value]
                    if len(set(value_lens)) > 1 and dtype is None:
                        # we have a ragged list so handle explicitly
                        value = as_tensor(
                            [np.asarray(val) for val in value], dtype=object
                        )
                return np.asarray(value, dtype=dtype)

            is_tensor = lambda x: isinstance(x, np.ndarray)

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)

                    # Removing this for now in favor of controlling the shape with `prepend_batch_axis`
                    # # at-least2d
                    # if tensor.ndim > 2:
                    #     tensor = tensor.squeeze(0)
                    # elif tensor.ndim < 2:
                    #     tensor = tensor[None, :]

                    self[key] = tensor
            except Exception as e:
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    ) from e
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding with"
                    " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"
                    f" features (`{key}` in this case) have excessive nesting (inputs type `list` where type `int` is"
                    " expected)."
                ) from e

        return self

    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        Send all values to device by calling `v.to(device)` (PyTorch only).

        Args:
            device (`str` or `torch.device`): The device to put the tensors on.

        Returns:
            [`BatchEncoding`]: The same instance after modification.
        """
        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        try:
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        except:
            print(
                f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported."
            )
        return self


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator


def _is_tensorflow(x):
    import tensorflow as tf

    return isinstance(x, tf.Tensor)


def is_tf_tensor(x):
    """
    Tests if `x` is a tensorflow tensor or not. Safe to call even if tensorflow is not installed.
    """
    return False if not _tf_available else _is_tensorflow(x)


def _is_torch(x):
    import torch

    return isinstance(x, torch.Tensor)


def _is_jax(x):
    import jax.numpy as jnp  # noqa: F811

    return isinstance(x, jnp.ndarray)


def is_jax_tensor(x):
    """
    Tests if `x` is a Jax tensor or not. Safe to call even if jax is not installed.
    """
    return False if not _flax_available else _is_jax(x)


def is_torch_tensor(x):
    """
    Tests if `x` is a torch tensor or not. Safe to call even if torch is not installed.
    """
    return False if not _torch_available else _is_torch(x)


def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    elif is_tf_tensor(obj):
        return obj.numpy().tolist()
    elif is_torch_tensor(obj):
        return obj.detach().cpu().tolist()
    elif is_jax_tensor(obj):
        return np.asarray(obj).tolist()
    elif isinstance(obj, (np.ndarray, np.number)):  # tolist also works on 0d np arrays
        return obj.tolist()
    else:
        return obj


from transformers import PreTrainedTokenizer
from .. import greedy_encoder
from ..greedy_encoder import build as build_greedy_encoder

_enums = {
    "only_first": greedy_encoder.TruncationStrategy.only_first,
    "only_second": greedy_encoder.TruncationStrategy.only_second,
    "longest_first": greedy_encoder.TruncationStrategy.longest_first,
    "do_not_truncate": greedy_encoder.TruncationStrategy.do_not_truncate,
    "longest": greedy_encoder.PaddingStrategy.longest,
    "max_length": greedy_encoder.PaddingStrategy.max_length,
    "do_not_pad": greedy_encoder.PaddingStrategy.do_not_pad,
    "truncate_right": greedy_encoder.TruncationSide.right,
    "truncate_left": greedy_encoder.TruncationSide.left,
    "pad_right": greedy_encoder.PaddingSide.right,
    "pad_left": greedy_encoder.PaddingSide.left,
}


def _splitter(text, pat):
    return regex.findall(pat, text)


# PushToHubMixin removed for now
class GreedTok(PreTrainedTokenizer):
    """ """

    vocab_files_names: Dict[str, str] = {}
    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {}
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}
    max_model_input_sizes: Dict[str, Optional[int]] = {}
    _auto_class: Optional[str] = None

    # first name has to correspond to main model input name
    # to make sure `tokenizer.pad(...)` works correctly
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]
    padding_side: str = "right"
    truncation_side: str = "right"

    def __init__(self, ranked_tokens=[], special_tokens_map={}, **kwargs):
        # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = copy.deepcopy(kwargs)
        # ranked_tokens should include special tokens
        # e.g. named_special_tokens_map = {'cls_token' : "<CLS>"}

        # self.special_tokens_map = special_tokens_map
        special_tokens_map = {} if special_tokens_map == None else special_tokens_map
        self.final_tokens_map = {}
        super().__init__(**kwargs)
        self.special_tokens = set(
            [
                (
                    special_token.encode("utf-8")
                    if isinstance(special_token, str)
                    else special_token
                )
                for special_token in special_tokens_map.values()
            ]
        )
        self.ranked_tokens = [
            r.encode("utf-8") if isinstance(r, str) else r for r in ranked_tokens
        ]
        for special_token in self.special_tokens:
            if special_token not in self.ranked_tokens:
                raise ValueError(f"{special_token} not included in ranked_tokens.")
        self.encoder = build_greedy_encoder(self.ranked_tokens, special_tokens_map)

        self.final_tokens = [
            self.encoder.get_rule(i) for i in range(self.encoder.get_rules_size())
        ]
        self.final_tokens_map = {k: i for i, k in enumerate(self.final_tokens)}
        self.final_ids_map = {i: k for k, i in self.final_tokens_map.items()}
        self.special_token_ids = set(
            [self.final_tokens_map[v] for v in self.special_tokens]
        )
        self.add_special_tokens(special_tokens_map)
        self.pattern = kwargs.get(
            "pattern",
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        )
        self.pat = regex.compile(self.pattern)

    def __len__(self) -> int:
        return len(self.final_tokens)

    def get_added_vocab(self):
        return self.final_tokens

    def _convert_token_to_id(self, token: bytes):
        if isinstance(token, str):
            return self.final_tokens_map[token.encode("utf-8")]
        return self.final_tokens_map[token]

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the
        vocab.

        Returns:
            `Dict[str, int]`: The vocabulary.
        """
        return self.final_tokens_map

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        **kwargs,
    ):
        r"""
        Instantiate a [`~tokenization_utils_base.PreTrainedTokenizerBase`] (or a derived class) from a predefined
        tokenizer.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                  user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - (**Deprecated**, not applicable to all derived classes) A path or url to a single saved vocabulary
                  file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g.,
                  `./my_model_directory/vocab.txt`.
            init_inputs (additional positional arguments, *optional*):
                Will be passed along to the Tokenizer `__init__` method.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the Tokenizer `__init__` method. Can be used to set special tokens like `bos_token`,
                `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,
                `additional_special_tokens`. See parameters in the `__init__` for more details.

        ```"""

        # At this point pretrained_model_name_or_path is either a directory or a model identifier name
        additional_files_names = {
            "added_tokens_file": ADDED_TOKENS_FILE,
            "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,
            "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
        }
        vocab_files = {**cls.vocab_files_names, **additional_files_names}
        resolved_vocab_files = {
            k: os.path.join(pretrained_model_name_or_path, v)
            for k, v in vocab_files.items()
        }
        init_configuration = {}

        return cls._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            **kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        *init_inputs,
        **kwargs,
    ):
        tokenizer_config_file = resolved_vocab_files.pop("tokenizer_config_file", None)
        if tokenizer_config_file is not None:
            with open(
                tokenizer_config_file, encoding="utf-8"
            ) as tokenizer_config_handle:
                init_kwargs = json.load(tokenizer_config_handle)
            # First attempt. We get tokenizer_class from tokenizer_config to check mismatch between tokenizers.
            config_tokenizer_class = init_kwargs.get("tokenizer_class")
            init_kwargs.pop("tokenizer_class", None)
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            config_tokenizer_class = None
            init_kwargs = init_configuration

        ranked_tokens = []
        added_tokens_file = resolved_vocab_files.pop("added_tokens_file", None)
        if added_tokens_file is not None:
            with open(added_tokens_file, encoding="utf-8") as handle:
                ranked_tokens = [
                    bytes.fromhex(t.strip()) for t in handle.read().strip().split("\n")
                ]

        special_tokens = []
        special_tokens_map_file = resolved_vocab_files.pop(
            "special_tokens_map_file", None
        )
        if special_tokens_map_file is not None:
            with open(
                special_tokens_map_file, encoding="utf-8"
            ) as special_tokens_map_handle:
                special_tokens_map = json.load(special_tokens_map_handle)
                for k, v in special_tokens_map.items():
                    special_tokens.append(v)

        return cls(ranked_tokens, special_tokens_map, *init_inputs, **init_kwargs)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        filename_prefix: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str]:
        """
        Save the full tokenizer state.


        This method make sure the full tokenizer can then be re-loaded using the
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] class method..

        Warning,None This won't save modifications you may have applied to the tokenizer after the instantiation (for
        instance, modifying `tokenizer.do_lower_case` after creation).

        Args:
            save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
            legacy_format (`bool`, *optional*):
                Only applicable for a fast tokenizer. If unset (default), will save the tokenizer in the unified JSON
                format as well as in legacy format if it exists, i.e. with tokenizer specific vocabulary and a separate
                added_tokens files.

                If `False`, will only save the tokenizer in the unified JSON format. This format is incompatible with
                "slow" tokenizers (not powered by the *tokenizers* library), so the tokenizer will not be able to be
                loaded in the corresponding "slow" tokenizer.

                If `True`, will save the tokenizer in legacy format. If the "slow" tokenizer doesn't exits, a value
                error is raised.
            filename_prefix (`str`, *optional*):
                A prefix to add to the names of the files saved by the tokenizer.
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.

        Returns:
            A tuple of `str`: The files saved.
        """
        if os.path.isfile(save_directory):
            raise OSError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)

        special_tokens_map_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + SPECIAL_TOKENS_MAP_FILE,
        )
        tokenizer_config_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_CONFIG_FILE,
        )
        tokenizer_config = copy.deepcopy(self.init_kwargs)
        added_tokens_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE,
        )

        # TODO: Ensure the modified attributes (those are also in the __init__ kwargs) will give identical tokenizers
        # target_keys = self.init_kwargs.keys()
        target_keys = ["model_max_length", "clean_up_tokenization_spaces"]
        for k in target_keys:
            if hasattr(self, k):
                tokenizer_config[k] = getattr(self, k)

        if len(self.init_inputs) > 0:
            tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)

        # Sanitize AddedTokens
        def convert_added_tokens(obj: Union[AddedToken, Any], add_type_field=True):
            if isinstance(obj, AddedToken):
                out = obj.__getstate__()
                if add_type_field:
                    out["__type"] = "AddedToken"
                return out
            elif isinstance(obj, (list, tuple)):
                return [
                    convert_added_tokens(o, add_type_field=add_type_field) for o in obj
                ]
            elif isinstance(obj, dict):
                return {
                    k: convert_added_tokens(v, add_type_field=add_type_field)
                    for k, v in obj.items()
                }
            return obj

        # add_type_field=True to allow dicts in the kwargs / differentiate from AddedToken serialization
        tokenizer_config = convert_added_tokens(tokenizer_config, add_type_field=True)

        # Add tokenizer class to the tokenizer config to be able to reload it with from_pretrained
        tokenizer_class = self.__class__.__name__
        # Remove the Fast at the end unless we have a special `PreTrainedTokenizerFast`

        tokenizer_config["tokenizer_class"] = tokenizer_class
        if getattr(self, "_auto_map", None) is not None:
            tokenizer_config["auto_map"] = self._auto_map
        if getattr(self, "_processor_class", None) is not None:
            tokenizer_config["processor_class"] = self._processor_class

        # remove private information
        if "name_or_path" in tokenizer_config:
            tokenizer_config.pop("name_or_path")
            tokenizer_config.pop("special_tokens_map_file", None)

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            out_str = (
                json.dumps(
                    tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False
                )
                + "\n"
            )
            f.write(out_str)
        print(f"tokenizer config file saved in {tokenizer_config_file}")

        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            out_str = (
                json.dumps(
                    self.special_tokens_map,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.write(out_str)
        print(f"special_tokens_map file saved in {special_tokens_map_file}")

        if self.final_tokens:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                for add in self.final_tokens:
                    f.write(add.hex() + "\n")
                print(f"added tokens file saved in {added_tokens_file}")

        # vocab_files = self.save_vocabulary(
        #     save_directory, filename_prefix=filename_prefix
        # )

        return (tokenizer_config_file, special_tokens_map_file, added_tokens_file)

    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        raise NotImplementedError

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        raise NotImplementedError

    def tokenize(
        self,
        text: str,
        pair: Optional[str] = None,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        Converts a string in a sequence of tokens, replacing unknown tokens with the `unk_token`.

        Args:
            text (`str`):
                The sequence to be encoded.
            pair (`str`, *optional*):
                A second sequence to be encoded with the first.
            add_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to add the special tokens associated with the corresponding model.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method. See details in
                [`~PreTrainedTokenizerBase.__call__`]

        Returns:
            `List[str]`: The list of tokens.
        """
        if not isinstance(text, str):
            raise TypeError("text is not a string.")
        encoding = self.encode(
            text, pair, add_special_tokens=add_special_tokens, **kwargs
        )
        if add_special_tokens:
            return [self.final_tokens_map[e] for e in encoding]
        else:
            return [
                self.final_tokens_map[e]
                for e in encoding
                if e not in self.special_token_ids
            ]

    @add_end_docstrings(
        """
            **kwargs: Passed along to the `.tokenize()` method.
        """,
        """
        Returns:
            `List[int]`, `torch.Tensor`, `tf.Tensor` or `np.ndarray`: The tokenized ids of the text.
        """,
    )
    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[int]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
        encoded_inputs = self._call_one(
            text,
            text_pair,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            max_length=max_length,
            stride=0,
            is_split_into_words=not isinstance(text, str),
            pad_to_multiple_of=1,
            return_attention_mask=False,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_token_type_ids=False,
            return_tensors=return_tensors,
            **kwargs,
        )

        if add_special_tokens:
            return encoded_inputs["input_ids"][0]
        else:
            return [
                x
                for x in encoded_inputs["input_ids"][0]
                if x not in self.special_token_ids
            ]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        raise NotImplementedError

    def _get_padding_truncation_strategies(
        self,
        padding=False,
        truncation=None,
        max_length=None,
        pad_to_multiple_of=None,
        verbose=True,
        **kwargs,
    ):
        raise NotImplementedError("Implemented in C++ backend")

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        text_pair: Optional[
            Union[
                TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
            ]
        ] = None,
        text_target: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        text_pair_target: Optional[
            Union[
                TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
            ]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            text_pair_target (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
                list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
                you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        """
        # To avoid duplicating
        all_kwargs = {
            "add_special_tokens": add_special_tokens,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "stride": stride,
            "is_split_into_words": is_split_into_words,
            "pad_to_multiple_of": pad_to_multiple_of,
            "return_tensors": return_tensors,
            "return_token_type_ids": return_token_type_ids,
            "return_attention_mask": return_attention_mask,
            "return_overflowing_tokens": return_overflowing_tokens,
            "return_special_tokens_mask": return_special_tokens_mask,
            "return_offsets_mapping": return_offsets_mapping,
            "return_length": return_length,
            "verbose": verbose,
        }
        all_kwargs.update(kwargs)
        if text is None and text_target is None:
            raise ValueError("You need to specify either `text` or `text_target`.")
        if text is not None:
            # The context manager will send the inputs as normal texts and not text_target, but we shouldn't change the
            # input mode in this case.\
            encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
        if text_target is not None:
            target_encodings = self._call_one(
                text=text_target, text_pair=text_pair_target, **all_kwargs
            )

        if text_target is None:
            return encodings
        elif text is None:
            return target_encodings
        else:
            encodings["labels"] = target_encodings["input_ids"]
            return encodings

    def _call_one(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ],
        text_pair: Optional[
            Union[
                TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
            ]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = 65535,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = 1,
        return_tensors: Optional[Union[str, TensorType]] = False,
        return_token_type_ids: Optional[bool] = False,
        return_attention_mask: Optional[bool] = False,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        is_batched: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # Input type checking for clearer error
        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return True
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(text):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str] ]` (batch of pretokenized examples)."
            )

        if text_pair is not None and not _is_valid_text_input(text_pair):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        if is_split_into_words:
            is_batched = (
                isinstance(text, (list, tuple))
                and text
                and isinstance(text[0], (list, tuple))
            )
        else:
            is_batched = isinstance(text, (list, tuple))

        if not is_batched:
            text = [text]
            text_pair = [text_pair] if text_pair != None else None

        if isinstance(text_pair, str):
            raise TypeError(
                "when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as"
                " `text`."
            )
        if text_pair is not None and len(text) != len(text_pair):
            raise ValueError(
                f"batch length of `text`: {len(text)} does not match batch length of `text_pair`:"
                f" {len(text_pair)}."
            )

        self.encoder.set_post_embedding_strategy(
            _enums["truncate_" + self.truncation_side],
            _enums["do_not_truncate"] if not truncation else _enums[truncation],
            _enums["pad_" + self.truncation_side],
            _enums["do_not_pad"] if not padding else _enums[padding],
            0 if max_length == None else max_length,
            1 if pad_to_multiple_of == None else pad_to_multiple_of,
        )

        def if_none_convert(x, value):
            return value if x == None else x

        if text_pair:
            # account for text pairs
            callback = kwargs.get(
                "callback",
                lambda x1, x2: (
                    [*x1, self.final_tokens_map[b" "], *x2],
                    [0] * len(x1) + [1] * (len(x2) + 1),
                ),
            )
            if is_split_into_words:
                encoded_inputs = self.encoder.batch_encode_pairs_presplit(
                    texts=text,
                    text_pairs=text_pair,
                    return_attention_mask=if_none_convert(return_attention_mask, False),
                    return_special_tokens_mask=if_none_convert(
                        return_special_tokens_mask, False
                    ),
                    return_token_type_ids=if_none_convert(return_token_type_ids, False),
                    stride=stride,
                    # f=lambda x1, x2: [*x1, *[self.final_tokens_map[b" "]], *x2]
                    f=callback,
                )
            else:
                encoded_inputs = self.encoder.batch_encode_pairs(
                    texts=text,
                    text_pairs=text_pair,
                    return_attention_mask=if_none_convert(return_attention_mask, False),
                    return_special_tokens_mask=if_none_convert(
                        return_special_tokens_mask, False
                    ),
                    return_token_type_ids=if_none_convert(return_token_type_ids, False),
                    stride=stride,
                    # f=lambda x1, x2: [*x1, *[self.final_tokens_map[b" "]], *x2]
                    f=callback,
                )
        else:
            callback = kwargs.get("callback", lambda x: x)
            if is_split_into_words:
                encoded_inputs = self.encoder.batch_encode_presplit(
                    texts=text,
                    return_attention_mask=if_none_convert(return_attention_mask, False),
                    return_overflowing_tokens=if_none_convert(
                        return_overflowing_tokens, False
                    ),
                    return_special_tokens_mask=if_none_convert(
                        return_special_tokens_mask, False
                    ),
                    stride=stride,
                    f=callback,
                )
            else:
                encoded_inputs = self.encoder.batch_encode(
                    texts=text,
                    return_attention_mask=if_none_convert(return_attention_mask, False),
                    return_overflowing_tokens=if_none_convert(
                        return_overflowing_tokens, False
                    ),
                    return_special_tokens_mask=if_none_convert(
                        return_special_tokens_mask, False
                    ),
                    stride=stride,
                    f=callback,
                )

        batch_outputs = BatchEncoding(
            encoded_inputs,
            tensor_type=return_tensors,
            prepend_batch_axis=True,
        )

        return batch_outputs

    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        raise NotImplementedError

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        raise NotImplementedError

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        raise NotImplementedError

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        raise NotImplementedError

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        raise NotImplementedError("Implemented in C++ backend")

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        raise NotImplementedError("Implemented in C++ backend")

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        raise NotImplementedError("Implemented in C++ backend")

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        NotImplementedError("Implemented in C++ backend")

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        raise NotImplementedError("Implemented in C++ backend")

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        raise NotImplementedError("Implemented in C++ backend")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        raise NotImplementedError

    def batch_decode(
        self,
        sequences: Union[
            List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"
        ],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
            for seq in sequences
        ]

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            return self.final_ids_map[token_ids].decode(
                "utf-8", errors="backslashreplace"
            )
        else:
            if skip_special_tokens:
                output = [
                    self.final_ids_map[x]
                    for x in token_ids
                    if x not in self.special_token_ids
                ]
            else:
                output = [self.final_ids_map[x] for x in token_ids]
            if clean_up_tokenization_spaces:
                output = self.clean_up_tokenization(output)
            return b"".join(output).decode("utf-8", errors="backslashreplace")

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        raise NotImplementedError

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (`str`): The text to clean up.

        Returns:
            `str`: The cleaned-up string.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    def _eventual_warn_about_too_long_sequence(
        self, ids: List[int], max_length: Optional[int], verbose: bool
    ):
        """
        Depending on the input and internal state we might trigger a warning about a sequence that is too long for its
        corresponding model

        Args:
            ids (`List[str]`): The ids produced by the tokenization
            max_length (`int`, *optional*): The max_length desired (does not trigger a warning if it is set)
            verbose (`bool`): Whether or not to print more information and warnings.

        """
        if max_length is None and len(ids) > self.model_max_length and verbose:
            if not self.deprecation_warnings.get(
                "sequence-length-is-longer-than-the-specified-maximum", False
            ):
                print(
                    "Token indices sequence length is longer than the specified maximum sequence length "
                    f"for this model ({len(ids)} > {self.model_max_length}). Running this sequence through the model "
                    "will result in indexing errors"
                )
            self.deprecation_warnings[
                "sequence-length-is-longer-than-the-specified-maximum"
            ] = True

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoTokenizer"):
        """
        Register this class with a given auto class. This should only be used for custom tokenizers as the ones in the
        library are already mapped with `AutoTokenizer`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoTokenizer"`):
                The auto class to register this new tokenizer with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        NotImplementedError

    def train_new_from_iterator(
        self,
        text_iterator,
        vocab_size,
        length=None,
        new_special_tokens=None,
        special_tokens_map=None,
        **kwargs,
    ):
        """
        Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
        as the current one.

        Args:
            text_iterator (generator of `List[str]`):
                The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts
                if you have everything in memory.
            vocab_size (`int`):
                The size of the vocabulary you want for your tokenizer.
            length (`int`, *optional*):
                The total number of sequences in the iterator. This is used to provide meaningful progress tracking
            new_special_tokens (list of `str` or `AddedToken`, *optional*):
                A list of new special tokens to add to the tokenizer you are training.
            special_tokens_map (`Dict[str, str]`, *optional*):
                If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special
                token name to new special token name in this argument.
            kwargs:
                Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.

        Returns:
            [`PreTrainedTokenizerFast`]: A new tokenizer of the same type as the original one, trained on
            `text_iterator`.

        """
        from ..pco_tokenizer import build as build_pco

        self.tokenizer = build_pco()

        max_token_length = kwargs.get("max_token_length", 100)
        min_word_count = kwargs.get("min_word_count", 1)
        pattern = kwargs.get("pattern", self.pat)

        special_tokens_map = {} if special_tokens_map == None else special_tokens_map
        pool = multiprocessing.Pool(kwargs.get("workers", 8))
        for b in text_iterator:
            if isinstance(b[0], str):  # splitting required
                # done in python because std::regex does not support pcre2 expressions such as \p{L}
                self.tokenizer.build_counter_from_text(
                    pool.map(partial(_splitter, pat=pattern), b)
                )
            else:
                self.tokenizer.build_counter_from_text(b)
        pool.close()
        self.tokenizer.initialize_graph(max_token_length, min_word_count)
        special_tokens_list = list(special_tokens_map.values())
        ranked_tokens, score = self.tokenizer.custom_steps(special_tokens_list)
        ranked_tokens, score = self.tokenizer.solve_to_step(vocab_size)
        print("len: ", len(ranked_tokens))
        return self.__class__(
            ranked_tokens=ranked_tokens,
            special_tokens_map=special_tokens_map,
            **kwargs,
        )

    def train_new_from_counts(
        self,
        word_counts,
        vocab_size,
        max_token_length=None,
        min_word_count=None,
        special_tokens_map=None,
        **kwargs,
    ):
        """
        Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline) as the current one.

        Args:
            word_counts (`Dict[str,int]`):
                The training corpus. Should be a dictionary of words to their respective counts
            vocab_size (`int`):
                The size of the vocabulary you want for your tokenizer.
            length (`int`, *optional*):
                The total number of sequences in the iterator. This is used to provide meaningful progress tracking
            new_special_tokens (list of `str` or `AddedToken`, *optional*):
                A list of new special tokens to add to the tokenizer you are training.
            special_tokens_map (`Dict[str, str]`, *optional*):
                If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special
                token name to new special token name in this argument.
            kwargs:
                Additional keyword arguments passed along to the trainer from the 🤗 Tokenizers library.

        Returns:
            [`PreTrainedTokenizerFast`]: A new tokenizer of the same type as the original one, trained on
            `text_iterator`.

        """
        from ..pco_tokenizer import build as build_pco

        self.tokenizer = build_pco(word_counts)
        special_tokens_map = {} if special_tokens_map == None else special_tokens_map
        self.tokenizer.initialize_graph(max_token_length, min_word_count)
        special_tokens_list = list(special_tokens_map.values())
        ranked_tokens, score = self.tokenizer.custom_steps(special_tokens_list)
        ranked_tokens, score = self.tokenizer.solve_to_step(vocab_size)
        print("len: ", len(ranked_tokens))
        return self.__class__(
            ranked_tokens=ranked_tokens,
            special_tokens_map=special_tokens_map,
            **kwargs,
        )


# To update the docstring, we need to copy the method, otherwise we change the original docstring.
# PreTrainedTokenizerBase.push_to_hub = copy_func(PreTrainedTokenizerBase.push_to_hub)
# if PreTrainedTokenizerBase.push_to_hub.__doc__ is not None:
#     PreTrainedTokenizerBase.push_to_hub.__doc__ = (
#         PreTrainedTokenizerBase.push_to_hub.__doc__.format(
#             object="tokenizer",
#             object_class="AutoTokenizer",
#             object_files="tokenizer files",
#         )
#     )
