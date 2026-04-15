# Model Results Report

## How To Read These Results

The main theme is alarm management: every threshold choice trades off nuisance alarms against missed or delayed detections.
For the SVM, the important values are false alarm rate, recall, and precision on held-out runs.
For the LSTM, the important values are false alarm rate, event recall, and detection delay, because early warning is a sequence problem rather than a single-row problem.

## SVM Interpretation

- The SVM was trained on 400 runs, tuned on 100 validation runs, and evaluated on 150 external runs.
- At the chosen threshold 0.1004, the external evaluation false alarm rate is 0.0394, recall is 0.4045, precision is 0.9938, and F1 is 0.5749.
- The default threshold produces positive-class recall 0.4583 and positive-class precision 0.9860 on the external evaluation split.
- The normal class recall is 0.8980, which means the model is fairly good at recognizing normal points when they appear, but the dataset is still heavily dominated by faulty rows.
- On the run-aware validation split, a threshold near 0.1004 keeps the false alarm rate around 0.0417 while recall is 0.5293.
- Interpretation: the SVM is a conservative detector. It can keep nuisance alarms low, but lowering false alarms also reduces the fraction of faulty rows that get detected.

## LSTM Interpretation

- The LSTM was trained on 300 runs and evaluated on 150 runs using windows of length 30 and stride 5.
- At the chosen threshold 0.6699, the false alarm rate is 0.0217, event recall is 0.9793, row F1 is 0.7832, and median detection delay is 25.0 samples.
- Near a false alarm rate of 0.0217, the LSTM still has event recall 0.9793 with median delay 25.0 samples.
- Negative delay values at very low thresholds mean the model is alarming before the assumed fault-onset time. That can be useful as a warning sign, but it also creates nuisance alarms if the threshold is too loose.
- Interpretation: the LSTM output should be read as an operating curve. Lower thresholds detect faults earlier and more often, while higher thresholds are quieter but slower.

## Multiclass SVM Interpretation

- The multiclass SVM was trained on 400 runs and evaluated on 150 runs across 21 classes.
- Overall multiclass accuracy is 0.3875, weighted F1 is 0.3530, and macro F1 is 0.3813.
- The best-recalled classes are class 4 (0.8383), class 7 (0.8349), class 6 (0.8333), class 5 (0.8319), class 1 (0.8270).
- The hardest classes are class 16 (0.0098), class 14 (0.0146), class 19 (0.0176), class 11 (0.0191), class 9 (0.0387).
- Interpretation: the multiclass SVM can identify some fault types reasonably well, but performance is very uneven across classes. This suggests that some faults have distinct static signatures while others overlap heavily in the 52-variable snapshot space.
- In practice, this means the multiclass output should be treated as a diagnostic aid rather than a uniformly reliable fault identifier across all 21 classes.

## Practical Takeaways

- Do not rely on accuracy alone. The threshold-specific alarm metrics are more informative for this project.
- A reasonable deployment-style choice is to start from a target false alarm budget, then choose the threshold that gives the best recall or event recall under that budget.
- For the SVM, that budget mostly controls the tradeoff between missed rows and nuisance alarms.
- For the LSTM, that budget also controls how early the alarm tends to fire.
- For multiclass classification, the per-class recall plot matters more than the overall accuracy because some fault labels are much easier than others.

## Generated Artifacts

- SVM plot: [svm_tradeoff.png](results/report/svm_tradeoff.png)
- LSTM tradeoff plot: [lstm_tradeoff.png](results/report/lstm_tradeoff.png)
- LSTM delay plot: [lstm_delay_curve.png](results/report/lstm_delay_curve.png)
- Multiclass recall plot: [svm_multiclass_recall.png](results/report/svm_multiclass_recall.png)