"""This module provides the tokenizer info class to handle the tokenizer information."""

from enum import Enum
from typing import List, Optional, Union

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from .base import XGRObject, _core


class VocabType(Enum):
    """The type of the vocabulary. Used in TokenizerInfo. XGrammar supports three types of
    vocabularies:

    RAW
        The vocabulary is in the raw format. The tokens in the vocabulary are kept in their
        original form without any processing. This kind of tokenizer includes the tiktoken
        tokenizer, e.g. microsoft/Phi-3-small-8k-instruct, Qwen/Qwen-7B-Chat, etc.

    BYTE_FALLBACK
        The vocabulary used in the byte fallback BPE tokenizer. The tokens are encoded through
        the byte-fallback conversion. E.g. "\u001B" -> "<0x1B>", " apple" -> "▁apple". This kind of
        tokenizer includes meta-llama/Llama-2-7b-chat, microsoft/Phi-3.5-mini-instruct, etc.

    BYTE_LEVEL
        The vocabulary used in the byte level BPE tokenizer. The tokens are encoded through
        the byte-to-unicode conversion, as in
        https://github.com/huggingface/transformers/blob/87be06ca77166e6a6215eee5a990ab9f07238a18/src/transformers/models/gpt2/tokenization_gpt2.py#L38-L59

        This kind of tokenizer includes meta-llama/Meta-Llama-3-8B-Instruct,
        meta-llama/Meta-Llama-3.1-8B-Instruct, etc.
    """

    RAW = "RAW"
    BYTE_FALLBACK = "BYTE_FALLBACK"
    BYTE_LEVEL = "BYTE_LEVEL"


class TokenizerInfo(XGRObject):
    """The tokenizer info contains the vocabulary, the type of the vocabulary, and necessary
    information for the grammar-guided generation.

    Note that although some tokenizers will encode the tokens in a special format, e.g.
    "<0x1B>" for "\u001B" in the ByteFallback tokenizer, and "Ġ" for " " in the Byte-Level BPE
    tokenizer, TokenizerInfo always decodes the vocabulary to the original format (e.g. "\u001B"
    and " ").

    Also note that some models (e.g. Phi-3 and Deepseek-V2) may pad the vocabulary to a multiple
    of 32. In this case, the model's vocab_size is larger than the tokenizer's vocabulary size.
    Please pass the model's vocab_size to the vocab_size parameter in the constructor, because
    this information is used to determine the size of the token mask.

    Parameters
    ----------
    encoded_vocab : Union[List[bytes], List[str]]
        The encoded vocabulary of the tokenizer.

    vocab_type : VocabType, default: VocabType.RAW
        The type of the vocabulary. See also VocabType.

    vocab_size : Optional[int], default: None
        The size of the vocabulary. If not provided, the vocabulary size will be len(encoded_vocab).

    stop_token_ids : Optional[List[int]], default: None
        The stop token ids. If not provided, the stop token ids will be auto detected (but may not
        be correct).

    prepend_space_in_tokenization : bool, default: False
        Whether the tokenizer will prepend a space before the text in the tokenization process.
    """

    def __init__(
        self,
        encoded_vocab: Union[List[bytes], List[str]],
        vocab_type: VocabType = VocabType.RAW,
        *,
        vocab_size: Optional[int] = None,
        stop_token_ids: Optional[Union[List[int], int]] = None,
        prepend_space_in_tokenization: bool = False,
    ) -> None:
        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]
        self._init_handle(
            _core.TokenizerInfo(
                encoded_vocab,
                vocab_type.value,
                vocab_size,
                stop_token_ids,
                prepend_space_in_tokenization,
            )
        )

    @staticmethod
    def from_huggingface(
        tokenizer: PreTrainedTokenizerBase,
        *,
        vocab_size: Optional[int] = None,
        stop_token_ids: Optional[Union[List[int], int]] = None,
    ) -> "TokenizerInfo":
        """Construct the tokenizer info from the huggingface tokenizer. This constructor supports
        various tokenizer backends, including the huggingface fast tokenizer and tiktoken tokenizer.
        Necessary information is automatically detected from the tokenizer.

        Note that some models (e.g. Phi-3 and Deepseek-V2) may pad the vocabulary to a multiple
        of 32. In this case, the model's vocab_size is larger than the tokenizer's vocabulary
        size. Please pass the model's vocab_size (this should be defined in the model config)
        to the vocab_size parameter in the constructor, because this information is used to
        determine the size of the token mask.

        Some models can have more than one stop token ids, and auto detection may not find all
        of them. In this case, you can specify the stop token ids manually.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizerBase
            The huggingface tokenizer.

        vocab_size : Optional[int], default: None
            The size of the vocabulary. If not provided, the vocabulary size will be
            len(encoded_vocab).

        stop_token_ids : Optional[List[int]], default: None
            The stop token ids. If not provided, the stop token ids will be auto detected
            (but may not be correct).

        Returns
        -------
        tokenizer_info : TokenizerInfo
            The tokenizer info.
        """

        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]
        if isinstance(stop_token_ids, list) and len(stop_token_ids) == 0:
            raise ValueError("stop_token_ids cannot be empty")

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
            # - the vocabulary is directly obtained from tokenizer.get_vocab()
            #   (tokenizer.backend_tokenizer.to_str() may not contain the full vocab, special
            #   tokens may be omitted)
            # - the vocab size is obtained from len(tokenizer.get_vocab()) or provided by user
            # - the vocab type and prepend_space_in_tokenization are obtained from
            #   tokenizer.backend_tokenizer.to_str()
            # - stop token id is provided by user, or auto detected.
            backend_str = tokenizer.backend_tokenizer.to_str()
            return TokenizerInfo._create_from_handle(
                _core.TokenizerInfo.from_huggingface(
                    encoded_vocab, backend_str, vocab_size, stop_token_ids
                )
            )
        elif (
            "vocab_file" in tokenizer.vocab_files_names
            and "tiktoken" in tokenizer.vocab_files_names["vocab_file"]
        ):
            # tiktoken tokenizer
            # e.g. Phi-3-small-8k-instruct, Qwen-7B-Chat, stablelm-2-12b-chat (previously)
            if stop_token_ids is None:
                if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                    stop_token_ids = [tokenizer.eos_token_id]
            return TokenizerInfo(
                encoded_vocab,
                VocabType.RAW,
                vocab_size=vocab_size,
                stop_token_ids=stop_token_ids,
                prepend_space_in_tokenization=False,
            )
        else:
            # TODO(yixin): sentencepiece tokenizer
            raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")

    @property
    def vocab_type(self) -> VocabType:
        """The type of the vocabulary."""
        return VocabType(self._handle.vocab_type)

    @property
    def vocab_size(self) -> int:
        """The size of the vocabulary."""
        return self._handle.vocab_size

    @property
    def prepend_space_in_tokenization(self) -> bool:
        """Whether the tokenizer will prepend a space before the text in the tokenization
        process."""
        return self._handle.prepend_space_in_tokenization

    @property
    def decoded_vocab(self) -> List[bytes]:
        """The decoded vocabulary of the tokenizer. This converts the tokens in the LLM's
        vocabulary back to the original format of the input text. E.g. for type ByteFallback,
        the token <0x1B> is converted back to "\u001B".
        """
        return self._handle.decoded_vocab

    @property
    def stop_token_ids(self) -> List[int]:
        """The stop token ids."""
        return self._handle.stop_token_ids

    @property
    def special_token_ids(self) -> List[int]:
        """The special token ids. Special tokens include control tokens, reserved tokens,
        padded tokens, etc. Now it is automatically detected from the vocabulary."""
        return self._handle.special_token_ids

    def dump_metadata(self) -> str:
        """Dump the metadata of the tokenizer to a json string. It can be used to construct the
        tokenizer info from the vocabulary and the metadata string."""
        return self._handle.dump_metadata()

    @staticmethod
    def from_vocab_and_metadata(
        encoded_vocab: List[Union[bytes, str]], metadata: str
    ) -> "TokenizerInfo":
        """Construct the tokenizer info from the vocabulary and the metadata string in json
        format.

        Parameters
        ----------
        encoded_vocab : List[Union[bytes, str]]
            The encoded vocabulary of the tokenizer.

        metadata : str
            The metadata string in json format.
        """
        return TokenizerInfo._create_from_handle(
            _core.TokenizerInfo.from_vocab_and_metadata(encoded_vocab, metadata),
        )
