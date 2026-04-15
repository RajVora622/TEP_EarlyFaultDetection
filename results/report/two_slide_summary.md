# Slide 1: Modeling Approach and Evaluation Setup

## Goal
- Build an alarm-management workflow for the Tennessee Eastman Process (TEP), not just a high-accuracy classifier.
- Study three tasks:
  - binary detection: `fault` vs `no fault`
  - multiclass diagnosis: predict `faultNumber`
  - early warning: detect faults quickly while controlling nuisance alarms

## What We Built
- Binary SVM baseline:
  - linear SVM on the 52 process variables
  - run-aware train/validation split
  - threshold tuned on held-out validation runs
- Multiclass SVM baseline:
  - same feature set
  - predicts one of 21 fault classes
- LSTM early detector:
  - sequence model using windows of 30 samples with stride 5
  - trained to raise alarms after assumed fault onset

## Why This Setup Matters
- TEP data is run-based, so splitting by run avoids overly optimistic results from correlated samples in the same run.
- Alarm systems should be judged by:
  - false alarm rate
  - recall / event recall
  - detection delay
- We renamed raw `xmeas_*` / `xmv_*` columns to readable process names to improve interpretability.

## Key Experimental Choices
- SVM threshold budget: target low nuisance alarms, using validation false alarm rate as the tuning constraint.
- LSTM onset assumptions:
  - training onset sample = 20
  - test onset sample = 160
- LSTM operating-point metrics:
  - event recall = was a faulty run detected at least once?
  - median detection delay = how many samples after onset was the fault detected?


# Slide 2: Results and Main Takeaways

## Binary SVM
- Chosen threshold: `0.1004`
- External evaluation:
  - false alarm rate = `0.0394`
  - recall = `0.4045`
  - precision = `0.9938`
  - F1 = `0.5749`
- Interpretation:
  - strong precision, so alarms are trustworthy
  - but recall is modest, so many fault samples are missed
  - best viewed as a conservative baseline detector

## Multiclass SVM
- Overall accuracy = `0.3875`
- Weighted F1 = `0.3530`
- Macro F1 = `0.3813`
- Best-recalled classes:
  - class 4 = `0.8383`
  - class 7 = `0.8349`
  - class 6 = `0.8333`
- Hardest classes:
  - class 16 = `0.0098`
  - class 14 = `0.0146`
  - class 19 = `0.0176`
- Interpretation:
  - some faults have clear static signatures
  - many fault classes overlap heavily, so diagnosis quality is uneven

## LSTM Early Detector
- Chosen threshold: `0.6699`
- Evaluation:
  - false alarm rate = `0.0217`
  - event recall = `0.9793`
  - row F1 = `0.7832`
  - median detection delay = `25` samples
- Interpretation:
  - nearly all faulty runs are detected
  - nuisance alarms remain low
  - threshold can be adjusted to trade earlier detection for more false alarms

## Bottom Line
- SVM is a solid baseline for conservative fault detection.
- Multiclass SVM shows that some faults are distinguishable, but many are not.
- LSTM is the strongest model for early warning because it captures time dynamics and supports a practical false alarm vs delay tradeoff.

## Suggested Visuals
- Binary SVM tradeoff: [svm_tradeoff.png](/Users/rajvora/Documents/CS567/TEP_Project/tep-fault-warning/results/report/svm_tradeoff.png)
- LSTM tradeoff: [lstm_tradeoff.png](/Users/rajvora/Documents/CS567/TEP_Project/tep-fault-warning/results/report/lstm_tradeoff.png)
- LSTM delay curve: [lstm_delay_curve.png](/Users/rajvora/Documents/CS567/TEP_Project/tep-fault-warning/results/report/lstm_delay_curve.png)
- Multiclass recall by class: [svm_multiclass_recall.png](/Users/rajvora/Documents/CS567/TEP_Project/tep-fault-warning/results/report/svm_multiclass_recall.png)
