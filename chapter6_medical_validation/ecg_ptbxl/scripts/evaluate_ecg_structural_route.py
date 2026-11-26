"""Technical ECG route-integrity benchmark (no clinical outcome labels).

Real verified PTB-XL inputs and actual trained checkpoints. The producer and
independent reference execute different implementations of the same XAI
methods. The detector receives observations and independently registered
contracts, never mutation labels. FuzzyXAI's diagnostic/repair API is used
unchanged. The ECG adapter and the controlled interventions are new.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from fuzzyxai.adapters.contracts_v2 import ExplanationContext
from fuzzyxai.adapters.system_source import derive_system_source_evidence
from fuzzyxai.diagnostics import (
    ActionableRepairPlanner, Contract, DiagnosticService,
    RepairExecutionContext, RouteEdge, RouteGraph, RouteNode
)
from fuzzyxai.diagnostics.repair_registry import ContractRepairProvider, RepairProviderRegistry
from fuzzyxai.system_semantics import build_system_evidence

from chapter6_medical_validation.ecg_ptbxl.src.data import LEADS
from chapter6_medical_validation.ecg_ptbxl.src.model import build_ecg_resnet1d
from chapter6_medical_validation.ecg_ptbxl.src.preprocessing import normalize
from chapter6_medical_validation.ecg_ptbxl.src.xai import temporal_occlusion
from chapter6_medical_validation.ecg_ptbxl.scripts.evaluate_retrain_independent import (
    independent_ig, independent_occlusion
)
from chapter6_medical_validation.shared.fuzzyxai_adapter import (
    SystemRunFacts, load_plan, observation_context
)
from chapter6_medical_validation.shared.hashing import sha256_file, sha256_json
from chapter6_medical_validation.shared.torch_adapter import TemperatureScaledTorchAdapter

ROOT = Path(__file__).resolve().parents[1]
LIMIT = .001
NODES = ("ig", "occlusion")
SHAPES = {"ig": (12,1000), "occlusion": (12,20)}
SCOPE = "controlled technical defects, not model errors or clinical safety"
META_FIELDS = ("object_id", "source_sha256", "checkpoint_sha256", "normalization_sha256",
               "standardized_input_sha256", "target", "lead_order", "method",
               "baseline", "output_space", "grid", "sampling_hz")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def array_digest(value):
    if value is None:
        return None
    a = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(str(a.shape).encode()+b"\0"+a.tobytes()).hexdigest()


def packet_item(meta, values):
    a = None if values is None else np.array(values, copy=True)
    return {"meta": copy.deepcopy(meta), "values": a, "declared_sha256": array_digest(a)}


def renew_digest(item):
    item["declared_sha256"] = array_digest(item["values"])


def rel_l1(value, reference, shape):
    if value is None:
        return None
    a, b = np.asarray(value, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        return None
    return float(np.abs(a-b).sum()/(np.abs(b).sum()+1e-6))


def graph_from_packet(packet, reference, route_id):
    nodes = [RouteNode("prediction","model_output","frozen_ecgresnet",None,{}, {},True,False,())]
    contracts = []
    edges = []
    for name in NODES:
        item = packet[name]
        ref = reference[name]
        expected = ref["meta"]
        obs = dict(item["meta"])
        value = item["values"]
        obs["shape"] = None if value is None else list(value.shape)
        obs["finite"] = None if value is None else bool(np.isfinite(value).all())
        obs["declared_sha256"] = item["declared_sha256"]
        obs["actual_sha256"] = array_digest(value)
        obs["map_distance"] = rel_l1(value, ref["values"], SHAPES[name])
        refs = (expected["source_sha256"], expected["checkpoint_sha256"], array_digest(ref["values"]))
        nodes.append(RouteNode(name,"explanation_artifact",name,"ptbxl-structural-v1",
                               expected, obs,True,True,refs))
        edges.append(RouteEdge("prediction_to_"+name,"prediction",name,"explains",True,{},{},False,(), "known_valid"))
        for field in META_FIELDS:
            contracts.append(Contract(f"{name}:{field}","equals",name,field,expected[field],
                                      category="explainer",evidence_refs=refs,source_nodes=(name,)))
        contracts.extend([
            Contract(f"{name}:shape","equals",name,"shape",list(SHAPES[name]),category="explainer",source_nodes=(name,)),
            Contract(f"{name}:finite","equals",name,"finite",True,category="explainer",source_nodes=(name,)),
            Contract(f"{name}:hash","checksum",name,"declared_sha256",obs["actual_sha256"],category="explainer",source_nodes=(name,)),
            Contract(f"{name}:reference_map","max_value",name,"map_distance",LIMIT,category="explainer",source_nodes=(name,))
        ])
    return RouteGraph(route_id,tuple(nodes),tuple(edges),tuple(contracts),
                      {"repair_costs":{"node:ig":.5,"node:occlusion":.5},
                       "contract_scope":"same-method independent replay with verified source and checkpoint"})


def direct_check(packet, reference):
    """Independent ordinary checks, exactly the same inputs and tolerances."""
    bad = []
    for name in NODES:
        item, ref = packet[name], reference[name]
        a = item["values"]
        d = rel_l1(a, ref["values"], SHAPES[name])
        if (any(item["meta"].get(k) != ref["meta"][k] for k in META_FIELDS)
            or a is None or a.shape != SHAPES[name] or not np.isfinite(a).all()
            or item["declared_sha256"] is None
            or item["declared_sha256"] != array_digest(a)
            or d is None or d > LIMIT):
            bad.append(name)
    return sorted(bad)


def corruptions(clean, independent, donor, extra):
    """Mutation generator is not visible to the detector or repair handlers."""
    yield "clean_native", (), copy.deepcopy(clean)
    yield "clean_independent_serialization", (), copy.deepcopy(independent)
    variants = []
    def new(kind, nodes):
        p = copy.deepcopy(clean)
        variants.append((kind,nodes,p))
        return p
    p = new("stale_other_ecg_ig",("ig",))
    p["ig"] = copy.deepcopy(donor["ig"])
    p = new("concealed_other_ecg_ig",("ig",))
    p["ig"]["values"] = donor["ig"]["values"].copy(); renew_digest(p["ig"])
    p = new("wrong_target",("ig",))
    p["ig"]["values"] = extra["other_target"].copy()
    p["ig"]["meta"]["target"] = 1-p["ig"]["meta"]["target"]; renew_digest(p["ig"])
    p = new("concealed_wrong_target",("ig",))
    p["ig"]["values"] = extra["other_target"].copy(); renew_digest(p["ig"])
    p = new("lead_permutation",("ig",))
    p["ig"]["values"] = p["ig"]["values"][[1,0,*range(2,12)],:]
    leads = list(p["ig"]["meta"]["lead_order"]); leads[0],leads[1]=leads[1],leads[0]
    p["ig"]["meta"]["lead_order"] = leads; renew_digest(p["ig"])
    p = new("concealed_lead_permutation",("ig",))
    p["ig"]["values"] = p["ig"]["values"][[1,0,*range(2,12)],:]; renew_digest(p["ig"])
    p = new("wrong_checkpoint_explanation",("ig",))
    p["ig"]["values"] = extra["other_checkpoint"].copy()
    p["ig"]["meta"]["checkpoint_sha256"] = extra["other_checkpoint_hash"]; renew_digest(p["ig"])
    p = new("wrong_preprocessing_explanation",("ig",))
    p["ig"]["values"] = extra["other_preprocessing"].copy()
    p["ig"]["meta"]["normalization_sha256"] = extra["other_normalization_sha"]
    p["ig"]["meta"]["standardized_input_sha256"] = extra["other_standardized_sha"]
    renew_digest(p["ig"])
    p = new("missing_normalization_trace",("ig",))
    p["ig"]["meta"].pop("normalization_sha256")
    p = new("self_consistent_map_scaling",("ig",))
    p["ig"]["values"] *= 2.; renew_digest(p["ig"])
    p = new("missing_map",("ig",))
    p["ig"]["values"] = None; renew_digest(p["ig"])
    p = new("stale_other_ecg_occlusion",("occlusion",))
    p["occlusion"] = copy.deepcopy(donor["occlusion"])
    p = new("compound_ig_occ",("ig","occlusion"))
    p["ig"]["values"] = donor["ig"]["values"].copy(); renew_digest(p["ig"])
    p["occlusion"]["meta"].pop("checkpoint_sha256")
    p = new("checksum_mismatch",("occlusion",))
    p["occlusion"]["declared_sha256"] = "0"*64
    yield from variants


def system_for(facts,pred,plan,validation):
    obs = observation_context(facts,plan).system_observation
    # The operator receives a derived verification result, not a mutation
    # flag or known fault class. Diagnosis is produced by the existing validator.
    issues = validation.issues
    bad = bool(issues)
    obs = replace(obs, trace_complete=not any(i.insufficient_evidence for i in issues),
                  trace_verification_source="DiagnosticService/actual ECG artifact verification",
                  rupture_present=bad,critical_rupture=any(i.severity=="error" for i in issues),
                  rupture_code="verified_ecg_route_contract_violation",
                  rupture_source_refs=tuple(i.violated_contract for i in issues))
    fp = facts.checkpoint_sha256
    src = derive_system_source_evidence(object_id=facts.object_id,model_fingerprint=fp,
        prediction=pred,internal_evidence={},source_interface_id=obs.alignment_transform.source_interface,
        risk_class=obs.risk_class,trace=obs.trace,model_trace=obs.model_trace,source_refs=obs.source_refs)
    return build_system_evidence(object_id=facts.object_id,model_fingerprint=fp,
                                 source=src,plan=plan,observation=obs)


def repair_packet(service, graph, report, packet, reference, unavailable=()):
    plan = report.repair_plan
    if plan is None:
        return graph, packet, None, []
    snapshots = {graph.trace_sha256:copy.deepcopy(packet)}
    operations = []
    def restore(current, step):
        name = step.target.subject_id
        if name not in NODES or name in unavailable:
            raise FileNotFoundError("registered reference unavailable for "+name)
        # Restores real, independently computed arrays and their verified
        # provenance; never simply writes 'expected' fields to the graph.
        updated = copy.deepcopy(snapshots[current.trace_sha256])
        updated[name] = copy.deepcopy(reference[name])
        new = graph_from_packet(updated,reference,graph.route_id)
        snapshots[new.trace_sha256] = updated
        operations.append(name)
        return new
    ctx = RepairExecutionContext(
        handlers={"restore_verified_ecg_explanation":restore},
        approved_step_ids=frozenset(s.step_id for s in plan.steps),
        allow_external_changes=True,
        satisfied_preconditions=frozenset(x for s in plan.steps for x in s.preconditions),
    )
    after, executions = service.repair_executor.execute(graph,plan,ctx)
    recert = service.recertifier.recertify(graph,after,plan,executions)
    return after, snapshots[after.trace_sha256], recert, operations


def metrics(rows, field):
    tp = sum(r["faulted"] and r[field] for r in rows)
    fp = sum(not r["faulted"] and r[field] for r in rows)
    fn = sum(r["faulted"] and not r[field] for r in rows)
    tn = sum(not r["faulted"] and not r[field] for r in rows)
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"recall":tp/max(1,tp+fn),
            "precision":tp/max(1,tp+fp),"false_alarm_rate":fp/max(1,fp+tn),
            "f1":2*tp/max(1,2*tp+fp+fn)}


def main():
    ap = argparse.ArgumentParser()
    for key in ("data-root","runs","out"):
        ap.add_argument("--"+key,type=Path,required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(int(os.environ.get("ECG_NUM_THREADS","4")))
    torch.manual_seed(20261126)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = args.data_root/"ecg"/"ptb-xl-1.0.3"/"prepared"
    manifest_path = prepared/"dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["all_43598_records100_files_verified"]
    for name, digest in manifest["prepared_sha256"].items():
        assert sha256_file(prepared/name)==digest, name
    signals = np.load(prepared/"signals.npy",mmap_mode="r",allow_pickle=False)
    ids = np.load(prepared/"ecg_ids.npy",allow_pickle=False)
    folds = np.load(prepared/"folds.npy",allow_pickle=False)
    stats = json.loads((prepared/"normalization.json").read_text())
    refs = json.loads((prepared/"source_pairs.json").read_text())
    patients = pd.read_csv(prepared/"label_construction.csv").set_index("ecg_id")["patient_id"]
    normal_sha = sha256_file(prepared/"normalization.json")
    runs = [(p.parent,json.loads(p.read_text())) for p in sorted(args.runs.rglob("run.json"))]
    assert len(runs)==3 and {r["seed"] for _,r in runs}=={2026,2027,2028}
    for path,r in runs:
        assert sha256_file(path/"best.pt")==r["checkpoint_sha256"]
        assert sha256_file(manifest_path)==r["execution"]["dataset_manifest_sha256"]
    chosen,run = min(runs,key=lambda pair:(min(x["validation_loss"] for x in pair[1]["history"]),-max(x["validation_auroc"] for x in pair[1]["history"]),pair[1]["seed"]))
    alt_path,alt = next((p,r) for p,r in runs if r["seed"]!=run["seed"])
    cfg = yaml.safe_load((ROOT/"configs"/"model_ecg_resnet1d.yaml").read_text())
    def load_model(path):
        m = build_ecg_resnet1d(tuple(cfg["channels"]),int(cfg["blocks_per_stage"])).to(device).eval()
        m.load_state_dict(torch.load(path/"best.pt",map_location=device,weights_only=True)["state_dict"])
        return m
    model,other_model = load_model(chosen),load_model(alt_path)
    temp = float(run["calibration"]["temperature"])
    def transform(value):
        return torch.from_numpy(normalize(np.asarray(value,dtype=np.float32).reshape(12,1000),stats))[None].to(device)
    adapter = TemperatureScaledTorchAdapter(model,temperature=temp,task="classification",ig_steps=64,input_transform=transform)
    plan = load_plan(ROOT/"configs"/"explain_plan_ecg.yaml")
    test = np.load(chosen/"test_predictions.npz",allow_pickle=False)
    idxs = test["prepared_indices"]
    assert len(idxs)==2107 and np.array_equal(idxs,np.flatnonzero(folds==10))
    probs = torch.softmax(torch.from_numpy(test["logits"])/temp,1).numpy()
    # Stratify by predicted class and confidence only; no diagnostic labels read.
    candidates = sorted(range(len(idxs)),key=lambda i:hashlib.sha256(f"20261126:{int(ids[idxs[i]])}".encode()).hexdigest())
    chosen_cases=[]; seen=set()
    for cls in (0,1):
        for high,n in ((True,16),(False,8)):
            picked=[]
            for i in candidates:
                index=int(idxs[i]); ecg_id=int(ids[index]); patient=int(patients.loc[ecg_id])
                if patient in seen or int(probs[i].argmax())!=cls or bool(probs[i].max()>=.9)!=high:
                    continue
                picked.append((i,index,ecg_id,patient));seen.add(patient)
                if len(picked)==n: break
            assert len(picked)==n,(cls,high,len(picked))
            chosen_cases.extend(picked)
    assert len(chosen_cases)==len(seen)==48
    dump(args.out/"cohort.json",{"records":[{"ecg_id":v[2],"prepared_index":v[1],"patient_hash":hashlib.sha256(f"ptbxl:{v[3]}".encode()).hexdigest(),"cached_prediction":int(probs[v[0]].argmax()),"cached_confidence":float(probs[v[0]].max())} for v in chosen_cases],"canonical_seed":run["seed"],"checkpoint_sha256":run["checkpoint_sha256"],"alternative_checkpoint_sha256":alt["checkpoint_sha256"],"dataset_manifest_sha256":sha256_file(manifest_path),"selection_no_labels":True,"protocol_sha256":sha256_file(ROOT/"configs"/"ecg_structural_route_protocol.json")})
    all_data=[]
    other_stats = copy.deepcopy(stats)
    other_stats["lead_mean"]=[float(x)+.15*float(s) for x,s in zip(stats["lead_mean"],stats["lead_std"])]
    for pos,index,ecg_id,patient in chosen_cases:
        raw=np.asarray(signals[index],dtype=np.float32); numeric=raw.reshape(1,-1)
        x=transform(numeric)
        pred=adapter.predict(numeric)
        p=tuple(float(x) for x in pred.probabilities[0]); target=int(np.argmax(p)); conf=max(p)
        assert np.max(np.abs(np.asarray(p)-probs[pos]))<1e-4 and target==int(probs[pos].argmax())
        ref=refs[str(ecg_id)]
        source_sha=ref["source_sha256"]
        patient_hash=hashlib.sha256(f"ptbxl:{patient}".encode()).hexdigest()
        facts=SystemRunFacts("ptb-xl-1.0.3-records100","test",f"ptbxl-{ecg_id}",source_sha,
            {"version":"ptbxl_train_lead_zscore_v1","lead_order":list(LEADS),"sample_rate_hz":100,"normalization_sha256":normal_sha},
            "ECGResNet1D",run["checkpoint_sha256"],run["run_id"],p,target,1,(patient_hash,*ref["files"].values()))
        native=adapter.extract_local_evidence(numeric,pred,ExplanationContext(target=target)).channels
        native_ig=np.asarray(native["integrated_gradients"],dtype=float).reshape(12,1000)
        native_occ=np.asarray(temporal_occlusion(model,x,target=target,temperature=temp,window=50,stride=50)["importance"],dtype=float)
        ig,ig_trace=independent_ig(model,x,target)
        occ=independent_occlusion(model,x,target,temp)
        alt_target,_=independent_ig(model,x,1-target)
        alt_checkpoint,_=independent_ig(other_model,x,target)
        alt_x=torch.from_numpy(normalize(raw,other_stats))[None].to(device)
        alt_preproc,_=independent_ig(model,alt_x,target)
        meta={"object_id":facts.object_id,"source_sha256":source_sha,"checkpoint_sha256":run["checkpoint_sha256"],
              "normalization_sha256":normal_sha,"standardized_input_sha256":array_digest(x.cpu().numpy()),"target":target,
              "lead_order":list(LEADS),"baseline":"zero_standardized_train_mean","sampling_hz":100}
        ig_meta={**meta,"method":"integrated_gradients","output_space":"logit","grid":[12,1000]}
        occ_meta={**meta,"method":"temporal_occlusion","output_space":"temperature_scaled_probability","grid":[12,20]}
        clean={"ig":packet_item(ig_meta,native_ig),"occlusion":packet_item(occ_meta,native_occ)}
        independent={"ig":packet_item(ig_meta,ig),"occlusion":packet_item(occ_meta,occ)}
        extra={"other_target":alt_target,"other_checkpoint":alt_checkpoint,"other_preprocessing":alt_preproc,
               "other_checkpoint_hash":alt["checkpoint_sha256"],"other_normalization_sha":sha256_json(other_stats),
               "other_standardized_sha":array_digest(alt_x.cpu().numpy())}
        evidence=args.out/"evidence"/f"ptbxl-{ecg_id}"; evidence.mkdir(parents=True)
        np.savez_compressed(evidence/"arrays.npz",waveform=raw,native_ig=native_ig,native_occ=native_occ,
            independent_ig=ig,independent_occ=occ,wrong_target_ig=alt_target,wrong_checkpoint_ig=alt_checkpoint,wrong_preproc_ig=alt_preproc)
        dump(evidence/"identity.json",{"ecg_id":ecg_id,"source_refs":ref,"patient_hash":patient_hash,"probabilities":p,
            "native_ig_completeness":native["ig_completeness"],"independent_ig_completeness":ig_trace,
            "metadata":{"ig":ig_meta,"occlusion":occ_meta},
            "native_independent_distances":{"ig":rel_l1(native_ig,ig,SHAPES["ig"]),"occlusion":rel_l1(native_occ,occ,SHAPES["occlusion"])},
            "alternative_source":{k:v for k,v in extra.items() if not isinstance(v,np.ndarray)}})
        all_data.append((ecg_id,raw,pred,facts,clean,independent,extra))
        print("REAL ECG",len(all_data),ecg_id,conf,flush=True)
    # Register only the domain-specific restore provider, without changing
    # the production solver, validator, planner, executor or recertifier.
    registry=RepairProviderRegistry(providers=[ContractRepairProvider(
        "ecg.restore_verified_artifact",frozenset(),frozenset({"explainer"}),
        "restore_verified_ecg_explanation","Restore verified ECG explanation artifact",1.0,False)])
    service=DiagnosticService(repair_planner=ActionableRepairPlanner(registry))
    rows=[]; negative=[]; examples=[]
    for i,(ecg_id,raw,pred,facts,clean,reference,extra) in enumerate(all_data):
        donor_id,_,_,_,donor,_,_=all_data[(i+1)%len(all_data)]
        assert donor_id!=ecg_id
        base_graph=graph_from_packet(reference,reference,f"base-{ecg_id}")
        base_validation=service.validator.validate(base_graph)
        assert base_validation.valid
        base_system=system_for(facts,pred,plan,base_validation)
        for j,(kind,truth_nodes,packet) in enumerate(corruptions(clean,reference,donor,extra)):
            route_id=hashlib.sha256(f"{ecg_id}:{j}:20261126".encode()).hexdigest()[:24]
            graph=graph_from_packet(packet,reference,route_id)
            report=service.diagnose(route=graph,repair_mode="plan")
            validation=service.validator.validate(graph)
            localized=sorted({n for issue in report.issues for n in issue.source_nodes})
            direct_nodes=direct_check(packet,reference)
            sys=system_for(facts,pred,plan,validation)
            after,repaired,recert,operations=repair_packet(service,graph,report,packet,reference)
            post=service.validator.validate(after)
            after_sys=system_for(facts,pred,plan,post)
            repaired_direct=direct_check(repaired,reference)
            result={"ecg_id":ecg_id,"patient_hash":facts.source_refs[0],"route_id":route_id,"scenario":kind,
                "faulted":bool(truth_nodes),"truth_nodes":list(truth_nodes),"donor_ecg_id":donor_id if "other_ecg" in kind or kind=="compound_ig_occ" else None,
                "prediction":facts.predicted_class,"confidence":max(facts.probabilities),"p_abnormal":facts.probabilities[1],
                "diagnostic_detected":bool(report.issues),"direct_detected":bool(direct_nodes),"confidence_detected":max(facts.probabilities)<.9,
                "diagnostic_nodes":localized,"direct_nodes":direct_nodes,"exact_localization":localized==sorted(truth_nodes),
                "direct_exact_localization":direct_nodes==sorted(truth_nodes),
                "issues":[{"id":x.violated_contract,"code":x.code,"nodes":list(x.source_nodes)} for x in report.issues],
                "minimal_cut":list(report.minimal_cut.atom_keys) if report.minimal_cut else [],
                "cut_optimal":bool(report.minimal_cut.optimality_proven) if report.minimal_cut else None,
                "before_action":sys.risk.action,"before_rho":sys.risk.rho,"before_gamma":sys.alignment["gamma"],
                "critical_override":sys.risk.critical_override,"clean_action":base_system.risk.action,
                "after_action":after_sys.risk.action,"after_rho":after_sys.risk.rho,
                "recovery_status":recert.status if recert else "not_needed",
                "recertified":bool(recert and recert.status=="full_success" and post.valid and not repaired_direct),
                "after_contract_valid":post.valid,"restored_nodes":operations,
                "same_clean_action_after":after_sys.risk.action==base_system.risk.action,
                "before_trace_sha256":graph.trace_sha256,"after_trace_sha256":after.trace_sha256,
                "map_distances":{name:graph.node(name).observed_attributes["map_distance"] for name in NODES},
                "packet_digests":{name:{"declared":packet[name]["declared_sha256"],"actual":array_digest(packet[name]["values"])} for name in NODES}}
            rows.append(result)
            if i==0 and kind in ("clean_native","concealed_other_ecg_ig","wrong_checkpoint_explanation","compound_ig_occ"):
                dump(args.out/"worked_examples"/f"{kind}.json",{"before":graph.to_dict(),"after":after.to_dict(),"report":report.to_dict(),
                    "recertification":None if recert is None else recert.__dict__,"outcome":result})
            if kind=="compound_ig_occ":
                denied,_,rec,ops=repair_packet(service,graph,report,packet,reference,unavailable=("occlusion",))
                blocked=system_for(facts,pred,plan,service.validator.validate(denied)).risk.action
                negative.append({"ecg_id":ecg_id,"status":rec.status if rec else "no_plan","valid_after":service.validator.validate(denied).valid,
                                 "action_after":blocked,"restored_nodes":ops})
        final_pred=adapter.predict(raw.reshape(1,-1))
        assert tuple(float(v) for v in final_pred.probabilities[0])==facts.probabilities,"model prediction changed"
        print("CHECKED",i+1,"patients",len(rows),"routes",flush=True)
        dump(args.out/"route_results.json",rows)
        dump(args.out/"unavailable_source_controls.json",negative)
    assert len(rows)==48*16
    faults=[r for r in rows if r["faulted"]]; clean=[r for r in rows if not r["faulted"]]
    high=[r for r in rows if r["confidence"]>=.9]
    all_stats={k:metrics(rows,k) for k in ("diagnostic_detected","direct_detected","confidence_detected")}
    high_stats={k:metrics(high,k) for k in ("diagnostic_detected","direct_detected","confidence_detected")}
    by_type={}
    for kind in sorted({r["scenario"] for r in rows}):
        group=[r for r in rows if r["scenario"]==kind]
        by_type[kind]={"n":len(group),"faulted":group[0]["faulted"],"detected":sum(r["diagnostic_detected"] for r in group),
                       "direct_detected":sum(r["direct_detected"] for r in group),"exact_localization":sum(r["exact_localization"] for r in group),
                       "blocked":sum(r["before_action"]=="block" for r in group),"recertified":sum(r["recertified"] for r in group)}
    report={"scope":SCOPE,"patients":48,"routes":len(rows),"faulted_routes":len(faults),"benign_routes":len(clean),
            "high_confidence_routes":len(high),"metrics":all_stats,"high_confidence_metrics":high_stats,
            "exact_source_localization_faulted":sum(r["exact_localization"] for r in faults),
            "direct_exact_source_localization_faulted":sum(r["direct_exact_localization"] for r in faults),
            "critical_block_faulted":sum(r["before_action"]=="block" and r["critical_override"] for r in faults),
            "clean_action_preserved":sum(r["before_action"]==r["clean_action"] for r in clean),
            "recertified_faulted":sum(r["recertified"] for r in faults),
            "recovered_original_action":sum(r["recertified"] and r["same_clean_action_after"] for r in faults),
            "optimal_cut_proven_count":sum(bool(r["cut_optimal"]) for r in faults),
            "targeted_restore_operations":sum(len(r["restored_nodes"]) for r in faults),
            "full_rebuild_artifacts":2*len(faults),
            "direct_rule_targeted_artifacts":sum(len(r["direct_nodes"]) for r in faults),
            "unavailable_reference_controls":{"n":len(negative),"incorrectly_recertified":sum(r["status"]=="full_success" or r["valid_after"] for r in negative),
                "blocked":sum(r["action_after"]=="block" for r in negative)},
            "same_information_detection_disagreements":sum(r["diagnostic_detected"]!=r["direct_detected"] for r in rows),
            "all_model_predictions_unchanged":True,"by_scenario":by_type,
            "limits":"known injected defects; verified independently recomputed reference maps available; no inference of medical correctness or actual clinical outcomes; no superiority claim over same-information rules"}
    dump(args.out/"summary.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print("COMPLETE ECG ROUTE STRUCTURAL BENCHMARK",len(rows),flush=True)


if __name__=="__main__":
    main()
