from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from tokenizer import SmilesTokenizer
from train import SmilesTransformer


DATA_PATH = Path("data/train.txt")

MAX_SAMPLES = 10_000
MAX_LENGTH = 64

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2

MODEL_PATH = "smiles_transformer.pt"


def load_smiles(path: Path) -> list[str]:
    df = pd.read_csv(path)

    return (
        df["SMILES"]
        .dropna()
        .astype(str)
        .head(MAX_SAMPLES)
        .tolist()
    )


def generate_smiles(
    model,
    tokenizer,
    device,
    max_length=64,
):
    model.eval()

    tokens = ["<BOS>"]

    for _ in range(max_length):
        token_ids = [
            tokenizer.stoi[token]
            for token in tokens
        ]

        x = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            logits = model(x)

        next_token_logits = logits[0, -1]

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1,
        )

        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1,
        ).item()

        next_token = tokenizer.itos[next_token_id]

        if next_token == "<EOS>":
            break

        if next_token == "<PAD>":
            continue

        tokens.append(next_token)

    smiles = "".join(tokens[1:])

    return smiles


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    smiles_list = load_smiles(DATA_PATH)

    tokenizer = SmilesTokenizer(smiles_list)

    model = SmilesTransformer(
        vocab_size=len(tokenizer.tokens),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_length=MAX_LENGTH,
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    print("Model loaded")
    print()

    for i in range(10):
        smiles = generate_smiles(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        print(f"Generated {i+1}:")
        print(smiles)
        print()


if __name__ == "__main__":
    main()