"""Immutable third-round design, written before the new dataset is evaluated."""
import json, hashlib
from revision.run_study import ROOT, SEEDS, EPS, safe_json, PRIMARY_PAIRS

DEST=ROOT/'results'/'models'/'kbs_resubmission'

def freeze():
    DEST.mkdir(parents=True, exist_ok=True)
    config={
      'version':'observable-kernel-3.0',
      'selection':'MF-only exploratory labelled validation; 12 overlapping resampled episodes, not an independent confirmatory set',
      'selection_limit':'Historical MF/HAR outcomes have already been inspected. New Daily Sports dataset is not used for tuning.',
      'primary_budget':2., 'epsilon':safe_json(EPS), 'seeds':SEEDS,
      'mf_primary_pairs':PRIMARY_PAIRS,
      'representation':{'kernel':'Gaussian Nystrom', 'rank':64, 'bandwidth':'median public pairwise squared distance',
         'dimension':'8 for epsilon<=1; 32 otherwise, bounded by public sample count',
         'ridge':.001, 'observability_weight':'sqrt(max(0,1-public calibration disagreement/(2*public calibration energy)))',
         'public_fit_cap':320, 'public_calibration_cap':80, 'public_labels':'never consumed'},
      'release':{'query':'class feature sums and class proportions / fixed public N',
         'clip':1.,'sensitivity':'2/N', 'delta':1e-5, 'adjacency':'labelled record replace-one',
         'scope':'provided independent descriptor or window; not subject, raw stream or participation DP'},
      'prediction':{'temperature':.1,'graph_neighbours':10,'beta':.2,'steps':50},
      'certificate':{'alpha':.05,'meaning':'simultaneous agreement with clean same-pipeline predictions; not semantic correctness'},
      'new_daily_sports':{'classes':19,'physical_views':['torso','right_arm','left_arm','right_leg','left_leg'],
         'primary_directions':[['torso','right_arm'],['right_arm','torso'],['left_arm','right_leg'],['right_leg','left_arm']],
         'subject_roles':{'public_fit':1,'public_calibration':1,'source':2,'target_train':2,'target_test':2},
         'features':'row-local statistics and FFT bands, no cohort preprocessing'},
      'baseline_scope':'Explicit mathematical protocol-matched controls; literature-inspired adaptations are never labelled faithful original reproductions.',
      'statistics':'One Holm family over all predeclared primary epsilon=2 dataset-direction versus numerical main comparators; corrected resampling approximation; overlapping subject limitations explicit',
      'hardware':'RTX 4080; PyTorch 2.0.0 CUDA 11.8; GPU public eigendecomposition and linear algebra; CPU graph and metrics'}
    path=DEST/'frozen_config.json'; blob=json.dumps(config,ensure_ascii=False,indent=2)
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==safe_json(config), 'Frozen design changed'
    else:path.write_text(blob,encoding='utf-8')
    return config

if __name__=='__main__':
    freeze();print('Frozen configuration:',DEST/'frozen_config.json')
