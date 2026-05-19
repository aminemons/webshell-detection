import math
import zlib
import re
from collections import Counter


class StatisticalFeatures:

    NASTY_FUNCTIONS = [
        "eval", "base64_decode", "base64_encode", "str_rot13", "gzuncompress",
        "gzinflate", "gzdecode", "str_replace", "preg_replace", "preg_match",
        "preg_match_all", "ereg_replace", "eregi_replace", "exec", "shell_exec",
        "system", "passthru", "popen", "proc_open", "pcntl_exec", "curl_exec",
        "curl_multi_exec", "parse_str", "extract", "unserialize", "phpinfo",
        "assert", "create_function", "call_user_func", "call_user_func_array",
    ]

    def entropy(self, text: str) -> float:
        if not text:
            return 0.0
        freq = Counter(text)
        total = len(text)
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    def longest_word(self, text: str) -> int:
        tokens = re.split(r"[^a-zA-Z0-9_]", text)
        return max((len(t) for t in tokens if t), default=0)

    def index_of_coincidence(self, text: str) -> float:
        n = len(text)
        if n < 2:
            return 0.0
        freq = Counter(text)
        return sum(c * (c - 1) for c in freq.values()) / (n * (n - 1))

    def compression_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        raw = text.encode("utf-8", errors="replace")
        if len(raw) == 0:
            return 0.0
        compressed = zlib.compress(raw, level=9)
        return len(compressed) / len(raw)

    def signature(self, text: str) -> int:
        lower = text.lower()
        return sum(1 for fn in self.NASTY_FUNCTIONS if fn in lower)

    def extract(self, text: str) -> dict:
        return {
            "LanguageIC": self.index_of_coincidence(text),
            "Entropy": self.entropy(text),
            "LongestWord": self.longest_word(text),
            "Compression_ratio": self.compression_ratio(text),
            "Signature": self.signature(text),
        }
