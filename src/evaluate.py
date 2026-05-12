from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem

from tokenizer import SmilesTokenizer
from generate import generate_smiles
from train import SmilesTransformer


DATA_PATH = Path("data/train.txt")
MODEL_PATH = Path("smiles_transformer.pt")
RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "generated_smiles.csv"

MAX_SAMPLES = 10_000
MAX_LENGTH = 64

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2

NUM_GENERATE = 100


def load_train_smiles() -> list[str]:
    df = pd.read_csv(DATA_PATH)

    return (
        df["SMILES"]
        .dropna()
        .astype(str)
        .head(MAX_SAMPLES)
        .tolist()
    )


def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_smiles = load_train_smiles()
    train_set = set(train_smiles)

    tokenizer = SmilesTokenizer(train_smiles)

    model = SmilesTransformer(
        vocab_size=len(tokenizer.tokens),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_length=MAX_LENGTH,
    ).to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    rows = []

    for i in range(NUM_GENERATE):
        smiles = generate_smiles(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=MAX_LENGTH,
        )

        valid = is_valid_smiles(smiles)
        novel = smiles not in train_set

        rows.append(
            {
                "index": i + 1,
                "smiles": smiles,
                "valid": valid,
                "novel": novel,
            }
        )

    result_df = pd.DataFrame(rows)

    valid_df = result_df[result_df["valid"]]
    unique_valid = valid_df["smiles"].drop_duplicates()
    novel_valid = unique_valid[
        ~unique_valid.isin(train_set)
    ]

    result_df.to_csv(OUTPUT_PATH, index=False)

    print("Generated:", len(result_df))
    print("Valid:", len(valid_df))
    print("Unique valid:", len(unique_valid))
    print("Novel valid:", len(novel_valid))
    print()

    print("Validity:", len(valid_df) / len(result_df))
    print("Uniqueness:", len(unique_valid) / max(len(valid_df), 1))
    print("Novelty:", len(novel_valid) / max(len(unique_valid), 1))
    print()

    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()