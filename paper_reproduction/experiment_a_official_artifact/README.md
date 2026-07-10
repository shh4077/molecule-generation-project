# Janus-QUBO Experiment A Official Artifact Reproduction

This folder contains Figure 2/3/4-style data generated from the official Janus-QUBO checkpoint and official QUBO matrix.

## Source

- Autoencoder checkpoint: `artifacts/paper/recon_loss=0.0043.ckpt`
- QUBO matrix: `artifacts/paper/qubo_128.csv`
- Experiment root: `experiments/paper_artifact`
- Seeds: 42, 123, 2026

## Contents

- `scripts/`: script used to generate figure-ready data
- `figure_data/figure2/`: QED-SA landscape, Pareto front, Table 1 comparison, representative molecules
- `figure_data/figure3/`: latent bit contribution and pair interaction data
- `figure_data/figure4/`: property and substructure shift data
- `aggregate/`: seed-aggregated Table 1 and alignment summaries
- `metadata/`: QUBO convention and Zenodo metadata

## Preview Figures

![Figure 2a QED-SA landscape](figure_data/figure2/figure2a_qed_sa_landscape.png)

![Figure 2b Table 1 comparison](figure_data/figure2/figure2b_table1_paper_vs_repro.png)

![Figure 3 bit contribution scatter](figure_data/figure3/figure3b_bit_contribution_scatter.png)

![Figure 4 property shift heatmap](figure_data/figure4/figure4_property_shift_heatmap.png)
