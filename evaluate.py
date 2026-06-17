#!/usr/bin/env python
# coding: utf-8
"""
Combined MDD evaluation script.

Inputs (via command-line args):
    input_dir/ref/ground_truth.csv   — columns: 'canonical', 'transcript'
    input_dir/res/results.csv        — column:  'predict'

Output:
    output_dir/scores.txt            — custom score and component metrics

Custom metric:
    Score = 0.5 * F1 + 0.4 * (1 - DER) + 0.1 * (1 - PER)

Pipeline mirrors Align.py + ins_del_cor_sub_analysis.py exactly.
dic[key] slot mapping (ref_human=0..2, human_our=3..5, ref_our=6..8):
    arr[0]=ref_aligned,   arr[1]=human_aligned, arr[2]=op_rh
    arr[3]=human_aligned, arr[4]=our_aligned,   arr[5]=op_ho
    arr[6]=ref_aligned,   arr[7]=our_aligned,   arr[8]=op_ro
"""

import os
import sys
import csv


# ---------------------------------------------------------------------------
# Needleman-Wunsch aligner  (mirrors metric.py / evaluate.py)
# ---------------------------------------------------------------------------

def _align(seq1, seq2):
    GAP = -1; MATCH = 1; MISMATCH = -1

    n, m = len(seq1), len(seq2)
    score = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        score[i][0] = GAP * i
    for j in range(n + 1):
        score[0][j] = GAP * j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[j - 1] == seq2[i - 1]:
                s = MATCH
            elif seq1[j - 1] == "<eps>" or seq2[i - 1] == "<eps>":
                s = GAP
            else:
                s = MISMATCH
            score[i][j] = max(
                score[i - 1][j - 1] + s,
                score[i - 1][j]     + GAP,
                score[i][j - 1]     + GAP,
            )

    align1, align2 = [], []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[j - 1] == seq2[i - 1]:
            s = MATCH
        elif seq1[j - 1] == "<eps>" or seq2[i - 1] == "<eps>":
            s = GAP
        else:
            s = MISMATCH
        if score[i][j] == score[i - 1][j - 1] + s:
            align1.append(seq1[j - 1]); align2.append(seq2[i - 1])
            i -= 1; j -= 1
        elif score[i][j] == score[i][j - 1] + GAP:
            align1.append(seq1[j - 1]); align2.append("<eps>")
            j -= 1
        else:
            align1.append("<eps>"); align2.append(seq2[i - 1])
            i -= 1
    while j > 0:
        align1.append(seq1[j - 1]); align2.append("<eps>"); j -= 1
    while i > 0:
        align1.append("<eps>"); align2.append(seq2[i - 1]); i -= 1

    align1.reverse(); align2.reverse()
    return align1, align2


def _ops(aligned1, aligned2):
    ops = []
    for r, h in zip(aligned1, aligned2):
        if   r != "<eps>" and h == "<eps>": ops.append("D")
        elif r == "<eps>" and h != "<eps>": ops.append("I")
        elif r != h:                        ops.append("S")
        else:                               ops.append("C")
    return ops


def _align_pair(s1, s2):
    """Strip '*', split, align, return (aligned1, aligned2, ops)."""
    seq1 = s1.replace("*", "").split()
    seq2 = s2.replace("*", "").split()
    a1, a2 = _align(seq1, seq2)
    return a1, a2, _ops(a1, a2)


# ---------------------------------------------------------------------------
# PER helpers  (mirrors Correct_Rate / Accuracy from metric.py)
# ---------------------------------------------------------------------------

def _per_counts(transcript_tokens, predict_tokens):
    """
    Returns (ins_del_sub_errors, num_phonemes) for PER calculation.

    Mirrors the Accuracy() function: counts insertions + deletions +
    substitutions against the reference (transcript) length.
    """
    ref_al, hyp_al, ops = _align_pair(
        " ".join(transcript_tokens),
        " ".join(predict_tokens),
    )
    errors = sum(1 for op in ops if op in ("I", "D", "S"))
    num_phonemes = len(transcript_tokens)
    return errors, num_phonemes


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def _read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_score(ground_truth_path, results_path):
    gt  = _read_csv(ground_truth_path)
    res = _read_csv(results_path)

    assert "canonical"  in gt[0],  "ground_truth.csv must have a 'canonical' column"
    assert "transcript" in gt[0],  "ground_truth.csv must have a 'transcript' column"
    assert "predict"    in res[0], "results.csv must have a 'predict' column"

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------

    # PER (mirrors Accuracy in metric.py)
    total_per_errors   = 0
    total_phonemes     = 0

    # MDD confusion matrix buckets
    cor_cor  = cor_nocor  = 0
    sub_sub  = sub_sub1   = sub_nosub  = 0
    ins_ins  = ins_ins1   = ins_noins  = 0
    del_del  = del_del1   = del_nodel  = 0

    # ------------------------------------------------------------------
    # Per-row processing
    # ------------------------------------------------------------------

    for gt_row, res_row in zip(gt, res):
        ref_str   = gt_row["canonical"]
        human_str = gt_row["transcript"]
        our_str   = res_row["predict"]

        # ---- PER: transcript (human) vs predict (our) -----------------
        human_tokens   = human_str.replace("*", "").split()
        predict_tokens = our_str.replace("*", "").split()
        errors, n_ph   = _per_counts(human_tokens, predict_tokens)
        total_per_errors += errors
        total_phonemes   += n_ph

        # ---- Three alignments -----------------------------------------
        #   ref_human  → ref_seq,    human_seq,  op_rh   (slots 0,1,2)
        #   human_our  → human_seq2, our_seq2,   op_ho   (slots 3,4,5)
        #   ref_our    → ref_seq3,   our_seq3,   op_ro   (slots 6,7,8)
        ref_seq,    human_seq,  op_rh = _align_pair(ref_str,   human_str)
        human_seq2, our_seq2,   op_ho = _align_pair(human_str, our_str)
        ref_seq3,   our_seq3,   op_ro = _align_pair(ref_str,   our_str)

        # ---- Deletion detection  (mirrors del detection loop) ---------
        flag = 0
        for i in range(len(ref_seq)):
            if ref_seq[i] == "<eps>":
                continue
            while flag < len(ref_seq3) and ref_seq3[flag] == "<eps>":
                flag += 1
            if flag < len(ref_seq3) and ref_seq[i] == ref_seq3[flag]:
                if   op_rh[i] == "D" and op_ro[flag] == "D":
                    del_del   += 1
                elif op_rh[i] == "D" and op_ro[flag] != "D" and op_ro[flag] != "C":
                    del_del1  += 1
                elif op_rh[i] == "D" and op_ro[flag] != "D" and op_ro[flag] == "C":
                    del_nodel += 1
                flag += 1

        # ---- Correct / Sub / Ins detection  (mirrors cor/sub/ins loop) 
        flag = 0
        for i in range(len(human_seq)):
            if human_seq[i] == "<eps>":
                continue
            while flag < len(human_seq2) and human_seq2[flag] == "<eps>":
                flag += 1
            if flag < len(human_seq2) and human_seq[i] == human_seq2[flag]:

                if   op_rh[i] == "C" and op_ho[flag] == "C":
                    cor_cor   += 1
                elif op_rh[i] == "C" and op_ho[flag] != "C":
                    cor_nocor += 1

                if   op_rh[i] == "S" and op_ho[flag] == "C":
                    sub_sub   += 1
                elif op_rh[i] == "S" and op_ho[flag] != "C" and ref_seq[i] != our_seq2[flag]:
                    sub_sub1  += 1
                elif op_rh[i] == "S" and op_ho[flag] != "C" and ref_seq[i] == our_seq2[flag]:
                    sub_nosub += 1

                if   op_rh[i] == "I" and op_ho[flag] == "C":
                    ins_ins   += 1
                elif op_rh[i] == "I" and op_ho[flag] != "C" and op_ho[flag] != "D":
                    ins_ins1  += 1
                elif op_rh[i] == "I" and op_ho[flag] != "C" and op_ho[flag] == "D":
                    ins_noins += 1

                flag += 1

    # ------------------------------------------------------------------
    # PER
    # ------------------------------------------------------------------
    # PER = fraction of phonemes that are wrong (ins+del+sub / total)
    PER = (total_per_errors / total_phonemes) if total_phonemes > 0 else 0.0

    # ------------------------------------------------------------------
    # F1  (detection)
    # ------------------------------------------------------------------
    TR = sub_sub + sub_sub1 + del_del + del_del1 + ins_ins + ins_ins1   # True Rejection
    FR = cor_nocor                                                        # False Rejection
    FA = sub_nosub + ins_noins + del_nodel                               # False Acceptance

    precision = TR / (TR + FR) if (TR + FR) > 0 else 0.0
    recall    = TR / (TR + FA) if (TR + FA) > 0 else 0.0
    F1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # ------------------------------------------------------------------
    # DER  (diagnosis error rate)
    # ------------------------------------------------------------------
    Correct_Diag = sub_sub  + ins_ins  + del_del
    Error_Diag   = sub_sub1 + ins_ins1 + del_del1
    DER = (Error_Diag / (Correct_Diag + Error_Diag)
           if (Correct_Diag + Error_Diag) > 0 else 0.0)

    # ------------------------------------------------------------------
    # Custom composite score
    # ------------------------------------------------------------------
    Score = 0.5 * F1 + 0.4 * (1 - DER) + 0.1 * (1 - PER)

    return Score, F1, DER, PER


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <ground_truth.csv> <results.csv>")
        sys.exit(1)

    _, gt_path, results_path = sys.argv

    score, f1, der, per = compute_score(
        gt_path,
        results_path,
    )

    print(f"Score: {score:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"DER: {der:.6f}")
    print(f"PER: {per:.6f}")