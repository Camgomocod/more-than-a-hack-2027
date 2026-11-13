"""Diagnose possession-spell errors for a submission against the public GT."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402

GT_PATH = os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
DEFAULT_PRED = os.path.join(HERE, "results", "clean_ensemble.csv")
FPS = 25.0


def _fmt_span(start, stop):
    return "{:6d}-{:6d} ({:5.1f}s)".format(
        start, stop, (stop - start) / FPS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=DEFAULT_PRED)
    parser.add_argument("--gt", default=GT_PATH)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    gt = pm.label_codes(pm.read_label_csv(args.gt))
    pred = pm.label_codes(pm.read_label_csv(args.pred))
    runs = pm.valid_runs(gt)

    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)

    gt_best = {}
    pred_best = {}
    for iou, gt_index, pred_index in overlaps:
        if gt_index not in gt_best:
            gt_best[gt_index] = (iou, pred_index)
        if pred_index not in pred_best:
            pred_best[pred_index] = (iou, gt_index)

    gt_flags = pm.matched_gt_flags(overlaps, len(gt_spells), args.iou)
    pred_matched = set()
    for iou, gt_index, pred_index in overlaps:
        if iou < args.iou:
            break
        if gt_flags[gt_index] and pred_index not in pred_matched:
            pred_matched.add(pred_index)
            if gt_flags[gt_index]:
                pass

    print("GT possessions: {}  predicted: {}".format(
        len(gt_spells), len(pred_spells)))
    print("\nGT spells (best same-team prediction):")
    for index, (start, stop, code) in enumerate(gt_spells):
        best = gt_best.get(index)
        if best is None:
            print("  {:2d} {} {}  NO OVERLAP".format(
                index, pm.LABELS[code], _fmt_span(start, stop)))
            continue
        iou, pred_index = best
        p_start, p_stop, p_code = pred_spells[pred_index]
        mark = "MATCH" if iou >= args.iou else "MISS "
        print("  {:2d} {} {}  best {} IoU={:.3f} pred#{} {}".format(
            index, pm.LABELS[code], _fmt_span(start, stop), mark, iou,
            pred_index, _fmt_span(p_start, p_stop)))

    print("\nPredicted spells (best same-team GT):")
    for index, (start, stop, code) in enumerate(pred_spells):
        best = pred_best.get(index)
        if best is None:
            print("  {:2d} {} {}  NO OVERLAP".format(
                index, pm.LABELS[code], _fmt_span(start, stop)))
            continue
        iou, gt_index = best
        g_start, g_stop, g_code = gt_spells[gt_index]
        mark = "MATCH" if iou >= args.iou else "SPUR "
        print("  {:2d} {} {}  best {} IoU={:.3f} gt#{} {}".format(
            index, pm.LABELS[code], _fmt_span(start, stop), mark, iou,
            gt_index, _fmt_span(g_start, g_stop)))

    scores = pm.spell_scores(gt, pred, runs)
    print("\nSpell F1 by IoU:")
    for threshold, result in scores["per_iou"].items():
        print("  {:.1f}: P={:.3f} R={:.3f} F1={:.3f} tp={} fp={} fn={}".format(
            threshold, result["precision"], result["recall"], result["f1"],
            result["n_matched"], result["n_spurious"], result["n_missed"]))

    gt_trans = pm.transitions(gt, runs)
    pred_trans = pm.transitions(pred, runs)
    for tol in pm.TAUS:
        result = pm.transition_scores(gt_trans, pred_trans, tol)
        print("Transitions tol={}: P={:.3f} R={:.3f} F1={:.3f}".format(
            tol, result["precision"], result["recall"], result["f1"]))
        print("  missed:")
        for transition in result["missed"]:
            print("    frame {:6d} {} -> {}".format(
                transition[0], pm.LABELS[transition[1]],
                pm.LABELS[transition[2]]))
        print("  spurious:")
        for transition in result["spurious"]:
            print("    frame {:6d} {} -> {}".format(
                transition[0], pm.LABELS[transition[1]],
                pm.LABELS[transition[2]]))


if __name__ == "__main__":
    main()
