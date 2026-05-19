import pandas as pd
from .statistical import StatisticalFeatures
from .syntactical import SyntacticalFeatures
from .lexical import LexicalFeatures


class FeatureExtractor:

    def __init__(self):
        self._stat = StatisticalFeatures()
        self._syn = SyntacticalFeatures()
        self._lex = LexicalFeatures()

    def extract_one(self, text: str) -> dict:
        features = {}
        features.update(self._stat.extract(text))
        features.update(self._syn.extract(text))
        features.update(self._lex.extract(text))
        return features

    def extract_dataframe(self, df: pd.DataFrame, code_col: str = "sourcecode") -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            code = str(row[code_col]) if pd.notna(row[code_col]) else ""
            feat = self.extract_one(code)
            feat["filename"] = row.get("filename", "")
            feat["label"] = row.get("label", "")
            records.append(feat)
        return pd.DataFrame(records)
