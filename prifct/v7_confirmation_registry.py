"""Freeze complete independently selected configurations before holdout use.

The registry cannot be created from an incomplete baseline grid. Numerical
tie handling changes no measured values and is logged beside raw rankings.
"""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/models/eswa_fedkapt/original_v7_research'
MF=BASE/'multiple_features_development'
OUT=BASE/'confirmation_v7'
ROSTER=['FedKAPT','DP-HP','DP-NTK','DP-KIP','KPG-RL','m-PPOT','UniOT','UniJDOT','AGW','UCOOT']
FILES={
    'private_DP_HP':2880,'private_DP_NTK':1920,'private_DP_KIP':2880,'private_DP_KIP_GDP':2880,
    'context_baselines':360,'structural_baselines':420,'structural_map_alternatives':1260,
}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def rank(values):return sorted(values,key=lambda r:(-round(r['accuracy'],12),-round(r['macro_f1'],12),r['nll']))


def main():
    summaries={};pending=[]
    for folder,expected in FILES.items():
        path=MF/folder/'summary.json'
        if not path.exists():pending.append(folder);continue
        report=json.loads(path.read_text())
        if report['rows']!=expected:pending.append(folder);continue
        rows=[json.loads(v) for v in (path.parent/'results.jsonl').read_text().splitlines()]
        assert len(rows)==expected and len({r['id'] for r in rows})==expected
        # Failed configurations remain visible and cannot be survivor averaged.
        summaries[folder]=report
    for dataset,expected in [('MF',26460),('DSA',17640)]:
        path=BASE/'core_mix_refinement'/dataset/'summary.json'
        if not path.exists():pending.append('core '+dataset);continue
        report=json.loads(path.read_text());assert report['rows']==expected and report['failures']==0
        summaries['core '+dataset]=report
    if pending:raise RuntimeError('Complete development first: '+', '.join(pending))
    selections={};ties=[]
    def choose(name,folder,candidates):
        assert candidates,name
        best=rank(candidates)[0]
        ties.append(dict(method=name,folder=folder,original_best=candidates[0],selected=best,
            same_primary_within_numerical_tolerance=abs(candidates[0]['accuracy']-best['accuracy'])<=1e-12,
            values_unchanged=True))
        return dict(folder=folder,options=best['options'],development=best)
    selections['FedKAPT']=choose('FedKAPT','../core_mix_refinement/MF',summaries['core MF']['rankings']['FedKAPT-candidate'])
    for name,folder in [('DP-HP','private_DP_HP'),('DP-NTK','private_DP_NTK')]:
        selections[name]=choose(name,folder,summaries[folder]['rankings'])
    kip=[]
    for folder in ['private_DP_KIP','private_DP_KIP_GDP']:
        kip += [dict(r,origin_folder=folder) for r in summaries[folder]['rankings']]
    best=rank(kip)[0];selections['DP-KIP']=choose('DP-KIP',best['origin_folder'],summaries[best['origin_folder']]['rankings'])
    for name,internal in [('m-PPOT','PPOT'),('UniOT','UniOT'),('UniJDOT','UniJDOT')]:
        selections[name]=choose(name,'context_baselines',summaries['context_baselines']['rankings'][internal])
    for name in ['KPG-RL','AGW','UCOOT']:
        selections[name]=choose(name,'structural_map_alternatives',summaries['structural_map_alternatives']['rankings'][name])
    ablations={d:{m:rank(v)[0]['options'] for m,v in summaries['core '+d]['rankings'].items()} for d in ['MF','DSA']}
    direct={}
    for d in ['MF','DSA']:
        p=BASE/'direct_core_control'/d/'summary.json';report=json.loads(p.read_text());assert report['failures']==0
        direct[d]=choose('Direct '+d,'direct_core_control/'+d,report['rankings'])['options']
    evidence={str((MF/f/'summary.json').relative_to(ROOT)):sha(MF/f/'summary.json') for f in FILES}
    evidence.update({str((BASE/'core_mix_refinement'/d/'summary.json').relative_to(ROOT)):sha(BASE/'core_mix_refinement'/d/'summary.json') for d in ['MF','DSA']})
    for folder in ['private_DP_HP','private_DP_NTK','private_DP_KIP','private_DP_KIP_GDP']:
        for name in ['execution_source_reuse.json','execution_source_reuse.jsonl']:
            p=MF/folder/name
            if p.exists():evidence[str(p.relative_to(ROOT))]=sha(p)
    for name in ['plan.json','merge_audit.json']:
        p=MF/'context_baselines/execution_shards'/name
        if p.exists():evidence[str(p.relative_to(ROOT))]=sha(p)
    data_protocol=json.loads((MF/'data_protocol.json').read_text())
    for family,expected in data_protocol['raw_source_sha256'].items():
        p=ROOT/'data/multiple_features'/('mfeat-'+family)
        assert sha(p)==expected,'Original MF input changed'
        evidence[str(p.relative_to(ROOT))]=expected
    for study in ['spectral_contrast','semantic_transport']:
        for p in (MF/study/'public_maps').glob('*.pkl'):
            evidence[str(p.relative_to(ROOT))]=sha(p)
    frozen_code={p:sha(ROOT/p) for p in ['prifct/v7_confirmation_registry.py','prifct/v7_mf_reserved_confirmation.py',
        'prifct/v7_core_target.py','prifct/v7_receiver_core.py','prifct/v7_core_checkpoints.py','prifct/v7_spectral_contrast.py',
        'prifct/author_baselines_v6/unidabench_core.py','prifct/v7_dp_kip.py','prifct/v7_lossless_probability.py',
        'prifct/v7_wisdm_core_confirmation.py','prifct/v7_wisdm_private_confirmation.py','prifct/v7_wisdm_cohorts.py',
        'prifct/v7_streamed_private_features.py','prifct/v7_dp_kip_gdp.py','prifct/v7_source_fit_memoization.py',
        'prifct/v7_confirmation_statistics.py','prifct/v7_mf_context_shards.py','prifct/v7_large_cohort_kip.py',
        'prifct/v7_multiple_features.py','prifct/v7_mf_common_data.py','prifct/v7_core_plan_bound.py',
        'prifct/v7_public_quadratic.py','prifct/v7_public_bridge.py','prifct/v7_functional_bridge.py',
        'prifct/v7_semantic_transport.py','prifct/v7_anchored_checked.py','prifct/v7_anchored_distribution.py',
        'prifct/v7_dp_ntk.py','prifct/v6_private_generation.py','prifct/v7_generator_weighted.py',
        'revision/joint_release.py','core/privacy.py','prifct/v7_structural_map_alternatives.py',
        'prifct/v7_replay_external_confirmation.py','prifct/v6_development_pilot.py']}
    protocol=dict(scope='MF reserved-record confirmation; benchmark itself previously exposed; no pristine dataset claim',
        roster=ROSTER,selections=selections,ablations=ablations,direct=direct,
        reserved_rule='300 indices from original label-blind77001 permutation; no training, public-map fit or tuning on their scores',
        main_budgets=[2.,10.],delta=1e-5,
        information='4 full-record private methods at equal total budgets;6 NONPRIVATE extra-source context methods with no epsilon claim',
        repeated_context='Nonprivate predictions computed once, not counted as two independent budget replicates',
        confirmation='Reuse fixed source/training-state checkpoints selected solely using development; source data and target adaptation unchanged',
        clustering='Retain deterministic original adaptation clustering, compute only new record-local responsibilities',
        statistics=dict(primary='Record accuracy averaged over all30 directions and private budgets; paired records shared across directions',
            bootstrap='10000 stratified paired record-ID resamples, identical draws across all directions/methods; fixed learned states/noise',
            independent_units='300 benchmark record identities, not30 directions or600 budget-record pairs; unavailable writer identities limit population inference',
            tests='Paired record-level sign-flip,100000 Monte Carlo draws, primary9 comparisons one Holm family;6 nonprivate contrasts contextual only',
            mechanism_tests='Separate8-comparison Holm family:6 independently tuned removals, independently tuned direct same-message control, released-count-only classifier with class functions erased',
            sign_flip_assumption='Conditional record-level sign symmetry of paired aggregate errors; record/writer dependence beyond known IDs not inferred',
            confidence='95% paired conditional bootstrap intervals; no source-cohort or DP-noise population claim',
            seed=77471),
        tolerance='Only12-decimal numerical ties in primary and macroF1; raw values/files unchanged',
        development_summary_sha256=evidence,frozen_code=frozen_code)
    OUT.mkdir(exist_ok=True);pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Confirmation registry immutable'
    else:
        pp.write_text(json.dumps(protocol,indent=2));(OUT/'protocol.sha256').write_text(sha(pp)+'\n')
        (OUT/'freeze_metadata.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),numerical_selection_audit=ties),indent=2))
    print(json.dumps(dict(protocol_sha256=sha(pp),methods=len(ROSTER),configuration_options={m:v['options'] for m,v in selections.items()})),flush=True)
    return protocol


if __name__=='__main__':main()
