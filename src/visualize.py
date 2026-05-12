from rdkit import Chem
from rdkit.Chem import Draw


SMILES = "COC(=O)c1c(C)n(C)c(O)c1-c1ccccc1C"


def main():
    mol = Chem.MolFromSmiles(SMILES)

    if mol is None:
        print("Invalid SMILES")
        return

    image = Draw.MolToImage(
        mol,
        size=(500, 500),
    )

    image.save("generated_molecule.png")

    print("Saved: generated_molecule.png")


if __name__ == "__main__":
    main()