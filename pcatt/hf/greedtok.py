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
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import multiprocessing
from functools import partial

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

from transformers.utils import TensorType, PaddingStrategy
from transformers.tokenization_utils_base import (
    BatchEncoding,
    TruncationStrategy,
    AddedToken,
)
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

        self.pattern = kwargs.get(
            "pattern",
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        )
        self.pat = regex.compile(self.pattern)
        self._in_target_context_manager = False
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
        self.add_special_tokens(special_tokens_map)

        self.encoder = build_greedy_encoder(self.ranked_tokens, special_tokens_map)
        self.final_tokens = [
            self.encoder.get_rule(i) for i in range(self.encoder.get_rules_size())
        ]
        self.final_tokens_map = {k: i for i, k in enumerate(self.final_tokens)}
        self.final_ids_map = {i: k for k, i in self.final_tokens_map.items()}

        # FIX special tokens out of order (credit: Shawn Tan)
        new_pairs = []
        for idx in list(self._added_tokens_decoder.keys()):
            token = self._added_tokens_decoder[idx].content
            new_pairs.append(
                (
                    self.final_tokens_map[token.encode("utf-8")],
                    self._added_tokens_decoder[idx],
                )
            )
            del self._added_tokens_decoder[idx]
        for idx, token_obj in new_pairs:
            self._added_tokens_decoder[idx] = token_obj
            self._added_tokens_encoder[token_obj.content] = self.final_tokens_map[
                token_obj.content.encode("utf-8")
            ]
            self._added_tokens_encoder[token_obj.content.encode("utf-8")] = idx

        self.special_token_ids = set(
            [self.final_tokens_map[v] for v in self.special_tokens]
        )

    def __len__(self) -> int:
        return len(self.final_tokens)

    @property
    def vocab_size(self) -> int:
        """
        `int`: Unlike other models, every token is special/non-special
        """
        raise len(self.final_tokens)

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
                Can be:
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
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

        init_kwargs.update(kwargs)
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
            return [self.final_ids_map[e] for e in encoding]
        else:
            return [
                self.final_ids_map[e]
                for e in encoding
                if e not in self.special_token_ids
            ]

    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        padding_side: Optional[bool] = None,
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
        Returns:
            `List[int]`, `torch.Tensor`, `tf.Tensor` or `np.ndarray`: The tokenized ids of the text.
        """

        encoded_inputs = self._call_one(
            text,
            text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=not isinstance(text, str),
            padding_side=padding_side,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"][0]

    def _init_set(self, key, current_value, value_if_key_not_exist):
        if current_value != None:
            return current_value
        elif key in self.init_kwargs:
            return self.init_kwargs[key]
        return value_if_key_not_exist

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
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: Optional[bool] = None,
        return_special_tokens_mask: Optional[bool] = None,
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

        padding = self._init_set("padding", padding, "do_not_pad")
        truncation = self._init_set("truncation", truncation, "do_not_truncate")
        max_length = min(
            self._init_set("max_length", max_length, 0),
            self._init_set("model_max_length", max_length, 0),
        )
        padding_strategy, truncation_strategy, max_length, _ = (
            self._get_padding_truncation_strategies(
                padding,
                truncation,
                max_length,
                pad_to_multiple_of,
                verbose=True,
                **kwargs,
            )
        )
        padding_side = self.padding_side if padding_side == None else padding_side
        return_token_type_ids = self._init_set(
            "return_token_type_ids", return_token_type_ids, False
        )
        return_attention_mask = self._init_set(
            "return_attention_mask", return_attention_mask, False
        )
        return_overflowing_tokens = self._init_set(
            "return_overflowing_tokens", return_overflowing_tokens, False
        )
        return_special_tokens_mask = self._init_set(
            "return_special_tokens_mask", return_special_tokens_mask, False
        )

        self.encoder.set_post_embedding_strategy(
            _enums["truncate_" + self.truncation_side],
            (
                _enums["do_not_truncate"]
                if not truncation_strategy
                else _enums[truncation_strategy.value]
            ),
            _enums["pad_" + padding_side],
            (
                _enums["do_not_pad"]
                if not padding_strategy
                else _enums[padding_strategy.value]
            ),
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

        if return_attention_mask and padding_strategy == PaddingStrategy.DO_NOT_PAD:
            encoded_inputs["attention_mask"] = [
                [1 for _ in range(len(ei))] for ei in encoded_inputs["input_ids"]
            ]

        batch_outputs = BatchEncoding(
            encoded_inputs,
            tensor_type=return_tensors,
            prepend_batch_axis=False,
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
        return b"".join([self.final_ids_map[x] for x in tokens]).decode(
            "utf-8", errors="backslashreplace"
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
            output = self.encoder.decode(token_ids, skip_special_tokens).decode(
                "utf-8", errors="backslashreplace"
            )
            if clean_up_tokenization_spaces:
                output = self.clean_up_tokenization(output)
            return output

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
        if isinstance(sequences[0], int):
            sequences = [sequences]
        return [
            (
                self.clean_up_tokenization(
                    seq.decode("utf-8", errors="backslashreplace")
                )
                if clean_up_tokenization_spaces
                else seq.decode("utf-8", errors="backslashreplace")
            )
            for seq in self.encoder.batch_decode(sequences, skip_special_tokens)
        ]

    def train_new_from_iterator(
        self,
        text_iterator,
        vocab_size,
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

        shortlist_size = kwargs.get("shortlist_size", 100)
        verbose = kwargs.get("verbose", True)
        self.tokenizer = build_pco({}, set(), shortlist_size, verbose)

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

        shortlist_size = kwargs.get("shortlist_size", 100)
        verbose = kwargs.get("verbose", True)
        self.tokenizer = build_pco(word_counts, set(), shortlist_size, verbose)
        shortlist_size = kwargs.get("shortlist_size", 100)
        verbose = kwargs.get("verbose", True)
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
