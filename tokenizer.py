"""
Tokenization Module for GPT Models

Includes:
- CharTokenizer: Character-level tokenizer for lightweight educational tasks
- BPETokenizer: Byte-Pair Encoding subword tokenizer from scratch with vocab training,
  merges table, and byte-level fallback for out-of-vocabulary robustness.
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class CharTokenizer:
    """
    Character-level tokenizer.
    
    Maps individual characters to unique integer IDs and vice-versa.
    """
    
    def __init__(self, vocab: Optional[List[str]] = None):
        self.vocab = vocab or []
        self.char_to_id: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_char: Dict[int, str] = {i: ch for i, ch in enumerate(self.vocab)}
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def fit(self, text: str) -> "CharTokenizer":
        """Build vocabulary from unique characters in raw text."""
        chars = sorted(list(set(text)))
        self.vocab = chars
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        return self
    
    def encode(self, text: str) -> List[int]:
        """Convert string to list of token IDs."""
        return [self.char_to_id.get(ch, 0) for ch in text]
    
    def decode(self, ids: Union[List[int], np.ndarray]) -> str:
        """Convert token IDs back to string."""
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return "".join([self.id_to_char.get(i, "") for i in ids])
    
    def save(self, filepath: str) -> None:
        """Save vocabulary to JSON."""
        data = {"type": "CharTokenizer", "vocab": self.vocab}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        """Load tokenizer from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"])


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Subword Tokenizer.
    
    Implements iterative frequency-based pair merging from scratch,
    similar to GPT-2 / SentencePiece byte-level BPE.
    """
    
    # Standard pre-tokenization regex pattern
    SPLIT_REGEX = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")
    
    def __init__(self, target_vocab_size: int = 1000, special_tokens: Optional[List[str]] = None):
        self.target_vocab_size = target_vocab_size
        self.special_tokens = special_tokens or ["<|endoftext|>", "<|pad|>", "<|unk|>"]
        self.merges: Dict[Tuple[bytes, bytes], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}
        self._init_base_vocab()
    
    def _init_base_vocab(self):
        """Initialize vocabulary with 256 individual bytes + special tokens."""
        self.vocab = {}
        self.inverse_vocab = {}
        
        # 1. Special tokens first
        for i, token in enumerate(self.special_tokens):
            token_bytes = token.encode("utf-8")
            self.vocab[i] = token_bytes
            self.inverse_vocab[token_bytes] = i
            
        # 2. Base 256 byte values
        offset = len(self.special_tokens)
        for b in range(256):
            token_bytes = bytes([b])
            token_id = offset + b
            self.vocab[token_id] = token_bytes
            self.inverse_vocab[token_bytes] = token_id
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def _get_stats(self, word_freqs: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
        """Compute frequency of adjacent byte token pairs."""
        pairs: Dict[Tuple[bytes, bytes], int] = {}
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs
    
    def _merge_pair(
        self,
        pair: Tuple[bytes, bytes],
        word_freqs: Dict[Tuple[bytes, ...], int]
    ) -> Dict[Tuple[bytes, ...], int]:
        """Merge all occurrences of a specific pair across vocabulary words."""
        new_word_freqs = {}
        first, second = pair
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs
    
    def train(self, text: str, min_freq: int = 2, verbose: bool = False) -> "BPETokenizer":
        """
        Train BPE tokenizer on text corpus until target_vocab_size is reached.
        """
        self._init_base_vocab()
        
        # Word frequency counting using whitespace & simple chunking
        words = text.split()
        word_freqs: Dict[Tuple[bytes, ...], int] = {}
        for w in words:
            # Prepend space to mimic GPT-2 byte encoding for token boundaries
            b_word = (" " + w).encode("utf-8")
            char_tuple = tuple(bytes([b]) for b in b_word)
            word_freqs[char_tuple] = word_freqs.get(char_tuple, 0) + 1
            
        num_merges = self.target_vocab_size - len(self.vocab)
        self.merges = {}
        
        for i in range(num_merges):
            stats = self._get_stats(word_freqs)
            if not stats:
                break
            
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < min_freq:
                break
                
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges[best_pair] = i
            
            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[new_id] = new_token
            self.inverse_vocab[new_token] = new_id
            
            if verbose and (i + 1) % 50 == 0:
                print(f"BPE merge {i + 1}/{num_merges}: {best_pair} (freq: {stats[best_pair]})")
                
        return self
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using learned BPE merge rules."""
        if not text:
            return []
            
        # Handle special tokens
        for s_token in self.special_tokens:
            if text == s_token:
                return [self.inverse_vocab[s_token.encode("utf-8")]]
                
        words = text.split()
        token_ids = []
        
        for idx, w in enumerate(words):
            prefix = " " if (idx > 0 or text.startswith(" ")) else ""
            b_word = (prefix + w).encode("utf-8")
            parts = [bytes([b]) for b in b_word]
            
            # Iteratively apply merges in rank order
            while len(parts) >= 2:
                # Find pair with lowest merge rank
                pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
                valid_pairs = [p for p in pairs if p in self.merges]
                if not valid_pairs:
                    break
                
                best_pair = min(valid_pairs, key=lambda p: self.merges[p])
                new_parts = []
                i = 0
                while i < len(parts):
                    if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                        new_parts.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_parts.append(parts[i])
                        i += 1
                parts = new_parts
            
            for part in parts:
                token_id = self.inverse_vocab.get(part, self.inverse_vocab.get(b"<|unk|>", 0))
                token_ids.append(token_id)
                
        return token_ids
    
    def decode(self, ids: Union[List[int], np.ndarray], errors: str = "replace") -> str:
        """Decode token IDs back to a UTF-8 string."""
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
            
        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                byte_chunks.append(b"")
                
        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors=errors)
    
    def save(self, filepath: str) -> None:
        """Save BPE merges and vocabulary to JSON."""
        serializable_merges = [
            [p[0].hex(), p[1].hex(), rank] for p, rank in self.merges.items()
        ]
        serializable_vocab = {
            str(k): v.hex() for k, v in self.vocab.items()
        }
        data = {
            "type": "BPETokenizer",
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self.special_tokens,
            "merges": serializable_merges,
            "vocab": serializable_vocab
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Load BPE tokenizer from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        tokenizer = cls(
            target_vocab_size=data["target_vocab_size"],
            special_tokens=data.get("special_tokens", ["<|endoftext|>", "<|pad|>", "<|unk|>"])
        )
        
        tokenizer.vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = {
            (bytes.fromhex(item[0]), bytes.fromhex(item[1])): item[2]
            for item in data["merges"]
        }
        return tokenizer
