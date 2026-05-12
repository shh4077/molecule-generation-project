from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import SmilesTokenizer


DATA_PATH = Path("data/train.txt")

MAX_SAMPLES = 10000
MAX_LENGTH = 64


def load_smiles(path: Path) -> list[str]:
    df = pd.read_csv(path)

    smiles_list = (
        df["SMILES"]
        .dropna()
        .astype(str)
        .head(MAX_SAMPLES)
        .tolist()
    )

    return smiles_list


class SmilesDataset(Dataset):
    def __init__(
        self,
        smiles_list: list[str],
        tokenizer: SmilesTokenizer,
        max_length: int,
    ):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.pad_token_id = tokenizer.stoi["<PAD>"]

    def __len__(self):
        return len(self.smiles_list)

    def pad_sequence(
        self,
        token_ids: list[int],
    ) -> list[int]:
        token_ids = token_ids[:self.max_length]

        padding_needed = (
            self.max_length
            - len(token_ids)
        )

        token_ids += (
            [self.pad_token_id]
            * padding_needed
        )

        return token_ids

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        token_ids = self.tokenizer.encode(smiles)

        token_ids = self.pad_sequence(token_ids)

        x = torch.tensor(
            token_ids[:-1],
            dtype=torch.long,
        )

        y = torch.tensor(
            token_ids[1:],
            dtype=torch.long,
        )

        return x, y


def main():
    smiles_list = load_smiles(DATA_PATH)

    tokenizer = SmilesTokenizer(smiles_list)

    dataset = SmilesDataset(
        smiles_list=smiles_list,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    x, y = next(iter(dataloader))

    print("Input shape:")
    print(x.shape)
    print()

    print("Target shape:")
    print(y.shape)
    print()

    print("Sample input:")
    print(x[0])
    print()

    print("Decoded sample:")
    print(
        tokenizer.decode(
            x[0].tolist()
        )
    )


if __name__ == "__main__":
    main()