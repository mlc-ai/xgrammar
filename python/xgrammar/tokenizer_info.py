# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The tokenizer info."""
from enum import Enum
from typing import List, Union

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from .base import XGObject, _core


class VocabType(Enum):
    """The type of the vocabulary. Used in TokenizerInfo. XGrammar supports three types of
    vocabularies:

    RAW
        The vocabulary is in the raw format. The tokens in the vocabulary is the same as the input
        string. This kind of tokenizer includes the tiktoken tokenizer, e.g.
        microsoft/Phi-3-small-8k-instruct, Qwen/Qwen-7B-Chat, etc.

    BYTE_FALLBACK
        The vocabulary used in the byte fallback BPE tokenizer. The tokens are processed through
        the byte-fallback conversion. E.g. "\u001B" -> "<0x1B>", " apple" -> "▁apple". This kind of
        tokenizer includes meta-llama/Llama-2-7b-chat, microsoft/Phi-3.5-mini-instruct, etc.

    BYTE_LEVEL
        The vocabulary used in the byte level BPE tokenizer. The tokens are processed through
        the byte-to-unicode conversion, as in
        https://github.com/huggingface/transformers/blob/87be06ca77166e6a6215eee5a990ab9f07238a18/src/transformers/models/gpt2/tokenization_gpt2.py#L38-L59

        This kind of tokenizer includes meta-llama/Meta-Llama-3-8B-Instruct,
        meta-llama/Meta-Llama-3.1-8B-Instruct, etc.
    """

    RAW = "RAW"
    BYTE_FALLBACK = "BYTE_FALLBACK"
    BYTE_LEVEL = "BYTE_LEVEL"


class TokenizerInfo(XGObject):
    """The tokenizer info, which contains the vocabulary, the type of the vocabulary, and necessary
    information for the grammar-guided generation. This class should be the first choice when
    handling tokenizers in XGrammar. It eliminates the overhead of converting the vocabulary between
    C++ and Python.

    Note that the vocabulary in TokenizerInfo is in the decoded format. Some tokenizers will encode
    the tokens in a special format. E.g. "<0x1B>" for "\u001B" in the ByteFallback tokenizer, and
    "Ġ" for " " in the Byte-Level BPE tokenizer. Huggingface tokenizer.get_vocab() will return
    the encoded vocabulary. TokenizerInfo will decode the vocabulary to the original format.

    Parameters
    ----------
    encoded_vocab : Union[List[bytes], List[str]]
        The vocabulary of the tokenizer.

    vocab_type : VocabType, default: VocabType.RAW
        The type of the vocabulary. See also VocabType.

    prepend_space_in_tokenization : bool, default: False
        Whether the tokenizer will prepend a space before the text in the tokenization process.
    """

    def __init__(
        self,
        encoded_vocab: Union[List[bytes], List[str]],
        vocab_type: VocabType = VocabType.RAW,
        prepend_space_in_tokenization: bool = False,
    ) -> None:
        self.init_with_handle(
            _core.TokenizerInfo(encoded_vocab, vocab_type.value, prepend_space_in_tokenization)
        )

    @property
    def vocab_type(self) -> VocabType:
        """The type of the vocabulary."""
        return VocabType(self.handle.vocab_type)

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary."""
        return self.handle.vocab_size

    @property
    def prepend_space_in_tokenization(self) -> bool:
        """Whether the tokenizer will prepend a space before the text in the tokenization
        process."""
        return self.handle.prepend_space_in_tokenization

    @property
    def decoded_vocab(self) -> List[bytes]:
        """The raw vocabulary of the tokenizer. This converts the tokens in the LLM's vocabulary
        back to the original format of the input text. E.g. for type ByteFallback, the token
        <0x1B> is converted back to "\u001B" in the raw vocabulary.
        """
        return self.handle.decoded_vocab

    @property
    def stop_token_ids(self) -> List[int]:
        """The stop token ids."""
        return self.handle.stop_token_ids

    @property
    def special_token_ids(self) -> List[int]:
        """The special token ids."""
        return self.handle.special_token_ids

    @staticmethod
    def from_huggingface(tokenizer: PreTrainedTokenizerBase) -> "TokenizerInfo":
        """Construct the tokenizer info from the huggingface tokenizer. This constructor supports
        various tokenizer backends, including the huggingface fast tokenizer and tiktoken tokenizer.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizerBase
            The huggingface tokenizer.

        Returns
        -------
        tokenizer_info : TokenizerInfo
            The tokenizer info.
        """

        try:
            encoded_vocab = tokenizer.get_vocab()
            encoded_vocab = [
                token for token, _ in sorted(encoded_vocab.items(), key=lambda x: x[1])
            ]
        except AttributeError as e:
            msg = (
                f"Cannot get the vocabulary of the tokenizer {type(tokenizer)}. The tokenizer "
                "should have a get_vocab method."
            )
            raise ValueError(msg) from e

        if isinstance(tokenizer, PreTrainedTokenizerFast):
            # huggingface fast tokenizer
            # Note this backend_str may not contain the full vocab. Some special tokens may
            # be omitted. So we still need to pass the vocab to the constructor.
            backend_str = tokenizer.backend_tokenizer.to_str()
            return TokenizerInfo.from_handle(
                _core.TokenizerInfo.from_huggingface(encoded_vocab, backend_str)
            )
        elif (
            "vocab_file" in tokenizer.vocab_files_names
            and "tiktoken" in tokenizer.vocab_files_names["vocab_file"]
        ):
            # tiktoken tokenizer
            # e.g. Phi-3-small-8k-instruct, Qwen-7B-Chat, stablelm-2-12b-chat (previously)
            return TokenizerInfo(encoded_vocab, VocabType.RAW, False)
        else:
            # TODO(yixin): sentencepiece tokenizer
            raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")

    def dump_metadata(self) -> str:
        """Dump the metadata of the tokenizer to a json string. It currently contains vocab_type
        and prepend_space_in_tokenization."""
        return self.handle.dump_metadata()

    @staticmethod
    def from_vocab_and_metadata(
        encoded_vocab: List[Union[bytes, str]], metadata: str
    ) -> "TokenizerInfo":
        """Construct the tokenizer info from the vocabulary and the metadata string in json format.

        Parameters
        ----------
        encoded_vocab : List[Union[bytes, str]]
            The vocabulary of the tokenizer.

        metadata : str
            The metadata string in json format.
        """
        return TokenizerInfo.from_handle(
            _core.TokenizerInfo.from_vocab_and_metadata(encoded_vocab, metadata),
        )
