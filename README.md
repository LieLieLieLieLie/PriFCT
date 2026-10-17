# PriFCT

Official research implementation of **Private Function-Consistent Transfer
(PriFCT)** for the Neurocomputing submission.

PriFCT performs feature-disjoint sensor transfer using unlabelled paired public
measurements and one differentially private class-moment/count release. The
receiver reconstructs class logits, converts them to probabilities, transfers
the probability function across sensor views, and applies an unlabelled prior
correction.

## Repository layout

- `prifct/`: PriFCT experiments, ablations, sensitivity analyses, and external
  validation runners.
- `core/`: shared preprocessing and privacy utilities.
- `revision/`: bounded-function construction and one-release mechanism.
- `requirements.txt`: Python dependencies.

Generated datasets, descriptors, predictions, checkpoints, and result ledgers
are intentionally excluded from the public repository. Scripts create `data/`
and `results/` locally as needed.

## Main entry points

Run commands from the repository root.

```bash
# Fixed UCI HAR comparison
python -m prifct.v47_uci_har proposed
python -m prifct.v47_uci_har private
python -m prifct.v47_uci_har context

# Same-packet ablation and sensitivity studies
python -m prifct.neurocomputing_r3_core_ablation
python -m prifct.neurocomputing_low_epsilon_private_baselines
python -m prifct.neurocomputing_streamlined_public_sensitivity
python -m prifct.neurocomputing_streamlined_fullmap_sensitivity
python -m prifct.neurocomputing_repeated_cohort_splits

# KU-HAR sequential external evaluation
python -m prifct.v48_ku_har
python -m prifct.v48_ku_har_analysis

# PAMAP2 sequential external evaluation
python -m prifct.v128_pamap2_acquisition
python -m prifct.v130_pamap2_preutility_feasibility
python -m prifct.v131_pamap2_descriptors
python -m prifct.v132_pamap2_frozen_prifct
python -m prifct.v133_pamap2_frozen_comparators
python -m prifct.v134_pamap2_independent_analysis
python -m prifct.v135_pamap2_gate_sensitivity
```

The private mechanism uses fixed seeds only for scientific replay. A real
deployment must generate fresh secret noise and must not publish its seed.

## Public data sources

The repository does not redistribute dataset records. Obtain them from the
official sources:

- [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [WISDM Smartphone and Smartwatch](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
- [selfBACK](https://archive.ics.uci.edu/dataset/521/selfback)
- [KU-HAR v5](https://doi.org/10.17632/45f952y38r.5)
- [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
- [Daily and Sports Activities](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities)
- [Multiple Features](https://archive.ics.uci.edu/dataset/72/multiple+features)

Follow each provider's licence and citation requirements. Dataset checksums and
participant-role manifests are generated or verified by the corresponding
acquisition scripts.

## Third-party baselines

Adapters reproduce published baselines under the common PriFCT protocol. Pinned
upstream baseline implementations are not relicensed as PriFCT code; obtain them
with the included acquisition scripts and follow their original licences.

## Availability and rights

This repository is public for peer review and reproducibility. It is not an
open-source licence grant. See [LICENSE](LICENSE): during submission and peer
review, copying, redistribution, mirroring, republication, sublicensing, or
incorporation into another work is prohibited without prior written permission.

