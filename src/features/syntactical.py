import re


class SyntacticalFeatures:

    def total_size(self, text: str) -> int:
        return len(text.encode("utf-8", errors="replace"))

    def nb_string(self, text: str) -> int:
        return len(re.findall(r'"[^"]*"|\'[^\']*\'', text))

    def nb_neq_string(self, text: str) -> int:
        return len(set(re.findall(r'"[^"]*"|\'[^\']*\'', text)))

    def nb_special_char(self, text: str) -> int:
        return sum(1 for c in text if not c.isalnum() and not c.isspace())

    def max_length_line(self, text: str) -> int:
        return max((len(line) for line in text.splitlines()), default=0)

    def nb_if(self, text: str) -> int:
        return len(re.findall(r"\bif\s*\(", text))

    def nb_loop(self, text: str) -> int:
        return len(re.findall(r"\b(?:for|while|foreach)\s*\(", text))

    def extract(self, text: str) -> dict:
        return {
            "total_size": self.total_size(text),
            "nb_string": self.nb_string(text),
            "nb_neq_string": self.nb_neq_string(text),
            "nb_special_char": self.nb_special_char(text),
            "max_length_line": self.max_length_line(text),
            "nb_if": self.nb_if(text),
            "nb_loop": self.nb_loop(text),
        }
