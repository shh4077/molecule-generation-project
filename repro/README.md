# Janus-QUBO Replication Pipeline

This folder contains local wrapper scripts for using your own `train.csv` and
`test.csv` SMILES data with the public Janus-QUBO code.

The goal is not exact paper reproduction, because the paper's 40M HelixDock +
100M Enamine training corpus is not included in the public artifacts. The goal
is a controlled replication:

1. train a Janus-style binary autoencoder on your 1.6M SMILES,
2. encode molecules into 128-bit latent codes,
3. train an FM surrogate and export a QUBO,
4. sample new latent codes,
5. decode them to SMILES,
6. compute Table 1 style metrics and compare them with the paper values.

## 0. Install

Use a GPU environment if possible.

```powershell
pip install -r external/Janus-QUBO/requirements.txt
pip install -r repro/requirements.txt
```

For a first smoke test, add `--limit 10000` to the preprocessing command.

## 1. CSV to Janus HDF5

```powershell
python repro/prepare_selfies_h5.py `
  --train-csv path/to/train.csv `
  --test-csv path/to/test.csv `
  --smiles-column smiles `
  --out-dir data/my_1p6m_block_bae `
  --vocab-mode paper `
  --max-len 96 `
  --dedupe
```

Outputs:

- `data/my_1p6m_block_bae/sequences.h5`
- `data/my_1p6m_block_bae/vocabulary.json`
- `data/my_1p6m_block_bae/clean_smiles.txt`
- `data/my_1p6m_block_bae/manifest.json`

## 2. Train the binary autoencoder

The upstream training script currently uses SwanLab logging and overrides
`max_epochs` internally. Keep that in mind when comparing runs.

```powershell
cd external/Janus-QUBO/Block_bAE
python train_gruencoder_transformerdecoder.py `
  --data_dir ../../../data/my_1p6m_block_bae `
  --save_dir ../../../runs/block_bae_my_1p6m `
  --accelerator gpu `
  --devices 1 `
  --batch_size 2048 `
  --num_workers 8 `
  --max_epochs 30 `
  --latent_dim 128 `
  --max_len 96
cd ../../..
```

The best checkpoint will be under `runs/block_bae_my_1p6m/checkpoints/`.

## 3. Encode your SMILES to 128-bit latent codes

```powershell
cd external/Janus-QUBO/Block_bAE
python encode_smiles.py `
  --ckpt_path ../../../runs/block_bae_my_1p6m/checkpoints/best.ckpt `
  --vocab_path ../../../data/my_1p6m_block_bae/vocabulary.json `
  --input_smiles_file ../../../data/my_1p6m_block_bae/clean_smiles.txt `
  --output_h5_path ../../../data/my_1p6m_latents.h5 `
  --max_len 96 `
  --batch_size 2048
cd ../../..
```

## 4. Train an FM surrogate and export QUBO

If your CSV has a real score column, use `--label-csv` and `--target-column`.
If not, the script can create a drug-likeness composite target from RDKit
properties.

```powershell
python repro/train_fm_qubo.py `
  --latent-h5 data/my_1p6m_latents.h5 `
  --output-dir runs/fm_my_1p6m `
  --target composite `
  --epochs 30 `
  --batch-size 8192
```

With your own score:

```powershell
python repro/train_fm_qubo.py `
  --latent-h5 data/my_1p6m_latents.h5 `
  --output-dir runs/fm_my_score `
  --label-csv path/to/train.csv `
  --label-smiles-column smiles `
  --target-column score
```

## 5. Sample QUBO latent codes

```powershell
python repro/sample_qubo.py `
  --qubo-csv runs/fm_my_1p6m/qubo_128.csv `
  --output-h5 runs/fm_my_1p6m/generated_latents.h5 `
  --n-samples 10000 `
  --n-chains 20000 `
  --sweeps 250
```

## 6. Decode generated latents to SMILES

```powershell
cd external/Janus-QUBO/Block_bAE
python latent_to_smiles.py `
  --input_h5 ../../../runs/fm_my_1p6m/generated_latents.h5 `
  --model_ckpt_path ../../../runs/block_bae_my_1p6m/checkpoints/best.ckpt `
  --vocab_path ../../../data/my_1p6m_block_bae/vocabulary.json `
  --output_h5 ../../../runs/fm_my_1p6m/generated_smiles.h5 `
  --device cuda:0
cd ../../..
```

## 7. Compare with paper Table 1

```powershell
python repro/evaluate_table1.py `
  --smiles runs/fm_my_1p6m/generated_smiles.h5 `
  --output-dir results/my_1p6m_vs_paper `
  --model-name My-Janus-FM `
  --paper-class Janus-FM `
  --ranking input `
  --deduplicate
```

If the generated file contains an explicit energy or score column, use a single
generator ranking:

```powershell
python repro/evaluate_table1.py `
  --smiles runs/fm_my_1p6m/generated_smiles.h5 `
  --output-dir results/my_1p6m_vs_paper_score_ranked `
  --model-name My-Janus-FM `
  --paper-class Janus-FM `
  --ranking score `
  --rank-column energies `
  --rank-order asc `
  --deduplicate
```

For an upper-bound diagnostic, rank separately for each metric:

```powershell
python repro/evaluate_table1.py `
  --smiles runs/fm_my_1p6m/generated_smiles.h5 `
  --output-dir results/my_1p6m_vs_paper_metric_oracle `
  --model-name My-Janus-FM `
  --paper-class Janus-FM `
  --ranking metric `
  --deduplicate
```

Important outputs:

- `results/my_1p6m_vs_paper/table1_comparison.csv`
- `results/my_1p6m_vs_paper/observed_table1.csv`
- `results/my_1p6m_vs_paper/generated_properties.csv`
- `results/my_1p6m_vs_paper/summary.json`

## Notes

- Use `--vocab-mode paper` for closest compatibility with the public checkpoint
  and architecture.
- Use `--vocab-mode build` only if many of your molecules are skipped because
  they contain SELFIES tokens not in the paper vocabulary.
- Exact Table 1 reproduction is not expected with a different 1.6M dataset.
  The useful result is the delta table: which metrics improve, collapse, or
  remain close under your data distribution.
