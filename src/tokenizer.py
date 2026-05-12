from pathlib import Path
import re

import pandas as pd


DATA_PATH = Path("data/train.txt")
MAX_SAMPLES = 10_000


SMILES_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|"
    r"[B-IK-Zb-ik-z]|\d|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?)"
)


def tokenize_smiles(smiles: str) -> list[str]:
    tokens = SMILES_PATTERN.findall(smiles)

    if "".join(tokens) != smiles:
        raise ValueError(f"Tokenization failed: {smiles}")

    return tokens


def load_smiles(path: Path) -> list[str]:
    df = pd.read_csv(path)

    return (
        df["SMILES"]
        .dropna()
        .astype(str)
        .head(MAX_SAMPLES)
        .tolist()
    )


class SmilesTokenizer:
    def __init__(self, smiles_list: list[str]):
        self.special_tokens = ["<PAD>", "<BOS>", "<EOS>"]

        vocab = set()

        for smiles in smiles_list:
            tokens = tokenize_smiles(smiles)
            vocab.update(tokens)

        self.tokens = self.special_tokens + sorted(vocab)

        self.stoi = {
            token: idx
            for idx, token in enumerate(self.tokens)
        }

        self.itos = {
            idx: token
            for token, idx in self.stoi.items()
        }

    def encode(self, smiles: str) -> list[int]:
        tokens = ["<BOS>"] + tokenize_smiles(smiles) + ["<EOS>"]

        return [
            self.stoi[token]
            for token in tokens
        ]

    def decode(self, token_ids: list[int]) -> str:
        tokens = [
            self.itos[idx]
            for idx in token_ids
        ]

        tokens = [
            token
            for token in tokens
            if token not in ["<PAD>", "<BOS>", "<EOS>"]
        ]

        return "".join(tokens)


def main() -> None:
    smiles_list = load_smiles(DATA_PATH)

    tokenizer = SmilesTokenizer(smiles_list)

    sample_smiles = smiles_list[0]
    encoded = tokenizer.encode(sample_smiles)
    decoded = tokenizer.decode(encoded)

    print("Number of SMILES used:", len(smiles_list))
    print("Vocabulary size:", len(tokenizer.tokens))
    print()

    print("Original:")
    print(sample_smiles)
    print()

    print("Tokens:")
    print(tokenize_smiles(sample_smiles))
    print()

    print("Encoded:")
    print(encoded)
    print()

    print("Decoded:")
    print(decoded)


if __name__ == "__main__":
    main()