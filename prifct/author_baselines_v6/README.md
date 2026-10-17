# Pinned author objective adapters: PPOT / UniOT / UniJDOT

This directory provides executable extra-resource NONPRIVATE contextual baselines. It does not certify FedKAPT utility, native published scores, or equal-information/privacy comparison. No real dataset utility was run while preparing this adapter.

`upstream/da3e5cfe02952d38f081e30a130d4660bc1e64fb` contains the unchanged 33-file UniDABench author snapshot, including the original MIT licence. `upstream_provenance.json` records the archive and each file SHA256. Every core import verifies the specific pinned file bytes before compiling the exact selected author AST definitions. PPOT/UniOT are the UniJDOT authors' time-series ports of those methods; they are not the original PPOT/UniOT authors' repository implementations.

The current Python environment is unchanged. Its original module import would require missing `easydl`, `skorch`, `S3` and `torchmetrics`. The wrapper bypasses unrelated trainer/backbone imports and supplies only the train/eval context and concatenating centroid accumulator needed by the selected author methods. No loss term is replaced. The original trainer and source files are preserved.

## Executable API

`unidabench_core.fit(method, source_x, source_y, target_x, num_classes, hparams, seed, hidden)` accepts only source labels and unlabelled target covariates, with fixed public common features supplied by the caller. A shared MLP maps these common features into the author heads. This is a disclosed public-map/common-feature architecture adaptation, not native raw-series CNN/FNO inference. Raw source 162 / raw target 108 feature arrays must first use the public bridge; they cannot be passed directly as corresponding coordinates.

The `hparams` values in this wrapper are explicit adapter defaults, not asserted to be the published best dataset-specific values. Native default grids from the pinned config files must also be included in final independent development tuning. PPOT's active author optimizer is hardcoded learning rate `.005`; the wrapper rejects an ineffective fake learning-rate sweep rather than silently pretending `hparams['learning_rate']` is used. A reviewed optimizer fix would be a new disclosed adapter version before scoring.

`predict` accepts covariates only, outputs normalized full-coverage posteriors plus a separate native rejection mask, and fixes the native adaptive-filling inference seed without altering the caller's RNG state. UniOT's rejection uses OT jointly across the supplied inference cohort; this must be disclosed as batch inference, not pointwise inductive screening. NLL/Brier/calibration metrics must not use zeroed rejected rows as probabilities.

`save_checkpoint` / `load_checkpoint` preserve all prediction buffers and unregistered state, including class memory, source prototypes, automatic threshold and beta. Restoring a trusted local checkpoint does not access data or retrain. The state and repeated raw source access are NONPRIVATE; DP privacy badges must never be attached to these baselines.

`context_runner.py` wraps the API for an NPZ with `source_x`, `source_y`, `adaptation_x`, `inference_x`, and separate role IDs; it rejects archives containing target labels and overlapping IDs. A JSON protocol must contain a frozen public-map SHA and explicitly preregistered hyperparameters. It produces saved predictions/checkpoint/provenance and computes no metrics. The caller must additionally verify subject-role disjointness and the actual public-map provenance; an asserted hash is not a proof.

## Completed checks

Run `static_audit.py` to recheck all 33 pinned source file hashes and generate the source-definition/dependency inventory. Run `smoke_unidabench.py` to train on entirely synthetic CUDA arrays only. All three objectives completed one source pretraining and one adaptation epoch on RTX 4080. Saved/reloaded posterior maximum differences were exactly zero; rejection masks also agreed. Native `evaluate` and the covariate-only decoder produced the same rejection masks and accepted probabilities within `1.2e-7`. Changing synthetic evaluation labels from all class 0 to all class 3 left native predictions unchanged.

The smoke does not show useful classification accuracy or reliable unknown detection. It only checks integration, label access and exact state replay. Actual role manifests, source/target sizes, development tuning and frozen final comparison are pending.

## Original trainer audit

The three algorithm training loops use source labels and ignore target labels for their losses. The native data loader nevertheless reads target labels to extend/reorder its encoder, derives private target class sets, constructs target few-shot evaluation subsets, and applies per-loaded-dataset standardization. The native sweep calculates target test metrics and target few-shot/test risks. These paths are not used by our wrapper. Its fixed last checkpoint replaces native optional best-source-risk selection, with an alias for the author's `Src_cls_loss` / `loss_cls` logging-key discrepancy. No target metric influences training, rejection threshold or selection.

## Faithfulness boundary for DP-summary adaptations

The transport solver subroutines can consume finite DP prototype supports as mathematical inputs. Their native algorithms additionally update learned source representations, source CE, moving prototypes, queues and/or source nearest neighbours from source records. A class mean cannot reconstruct that information. Deleting those terms is a separately named inspired summary comparator, not a faithful native PPOT/UniOT/UniJDOT reproduction. These three executions therefore remain contextual controls with explicit extra-resource badges.
