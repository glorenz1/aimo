# Recorded results

PEP corresponds to submission 994986 and NRC to submission 1023896.

The [official numeric extract](public-validation.json) records accuracy `0.64`,
coverage `1.0` and zero invalid predictions for both methods, Competition phase
30038, Small Models task 36241. These are previously consulted public-validation
scores. The author separately reported 9/14 correct for each. No final private-test result,
official runtime tie-break or final ranking is inferred from these records.

The [development table](../results/development.csv) summarizes the original
grouped evaluation. Brier values are rounded to four decimal places. The
33 binary Small Models rows span 15 texts and 14 inferred groups, with 11
non-robust and 22 robust rows. Four outer folds in each of three procedures
(seeds 2026/2027/2028) produce 99 dependent held-group occurrences. Auxiliary
training uses 21 Boolean-labeled public GPT rows; held-group aliases are excluded
from auxiliary fitting too. These groups were repeatedly reused for selection.

| Matched contribution | Accuracy gain (points) | Balanced accuracy gain (points) |
|---|---:|---:|
| PEP internal inputs vs metadata-only refit | 9.09 | 9.85 |
| NRC complete white-box group vs no-white-box refit | 8.08 | 11.36 |
| NRC regional additions vs numerical/alignment correction | 5.05 | 3.79 |

These comparisons overlap and cannot be added. PEP's internal inputs worsen
Brier/log loss in this comparison. NRC's complete white-box group improves both,
while its regional additions worsen them by 0.00646/0.02017. All five regional
corrections are robust cases from two groups.

The main finding concerns refitting: removing saved predictions leaves
the original accuracy contributions positive, while excluding groups and
retraining can reverse them. NRC's complete contribution reverses in 12/14
reselected/repartitioned exclusions; seven remain negative with inherited
partitions and fixed original regularization. Balanced and probability metrics
often retain benefits. These are retrospective sensitivity analyses, not
independent confirmation or a causal explanation of mathematical reasoning.

[Migration validation](validation.json) records 12 passing tests, exact final-head
parity on 33 authorized TRAIN cells, and all 99 held-group NRC decisions matching
with maximum probability difference 2.22e-16. Official ingestion with explicitly
substituted cached measurements reports zero invalid predictions for each method.
Fresh GPU inference, full fitting, the full sensitivity study and private
evaluation were not rerun for this migration.
