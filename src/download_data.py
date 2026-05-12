from pathlib import Path
import urllib.request

import pandas as pd
from rdkit import Chem


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "train.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "molecularsets/moses/master/data/train.csv"
)


def download_dataset() -> None:
    if DATA_PATH.exists():
        print("Dataset already exists")
        return

    print("Downloading dataset...")

    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

    print(f"Saved to: {DATA_PATH}")


def load_dataset() -> None:
    df = pd.read_csv(DATA_PATH)

    print()
    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print()

    smiles_column = (
        "SMILES"
        if "SMILES" in df.columns
        else df.columns[0]
    )

    print("Using SMILES column:", smiles_column)
    print()

    for smiles in df[smiles_column].head(5):
        mol = Chem.MolFromSmiles(smiles)

        print(smiles)
        print("Valid molecule:", mol is not None)
        print()


def main() -> None:
    download_dataset()
    load_dataset()


if __name__ == "__main__":
    main()