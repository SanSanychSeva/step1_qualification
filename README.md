# this project containes jupyter notebooks with solutions to Yandex ShAD Week Intensive admittance qualification exam
## event: AI Analytics Week Школы анализа данных (YSDA) in Nov-Dec 2026
## notebooks (in project subfolder "notebooks") list:

### task A: Mirror for Algernon
Researchers studied whether the presence of a mirror is associated with laboratory mice's choice of where to spend their time.

16 laboratory mice were individually placed in cages consisting of two rooms connected by a tunnel. One of the rooms contained a mirror, while the other did not. During the experiment, the position of the mirror was changed to reduce the influence of room location on the results.

For each mouse, the proportion of time spent in the room with the mirror was determined. The distribution of the resulting values is shown in the histogram.

![Mice-Mirror experiments results](figs/task_A_mice_histogram.png)

The horizontal axis shows the proportion of time spent in the room containing mirror, expressed as a percentage; the vertical axis shows the number of mice.

It is known that each mouse spent more time in one of the two rooms than in the other. Assume that the results for different mice are independent and that the probability of spending more time in the room without a mirror is the same for all mice.

Can we conclude at the `5%` significance level that mice tend to spend more time in one of the two rooms - with or without a mirror?

Propose a statistical test based on the available data, calculate the exact two-sided `p-value`, and draw a conclusion at the significance level $\alpha=0.05$. When calculating the `p-value`, account for deviations from the expected result in both the greater and the lesser directions.

#### Output format
In the answer field, enter two numbers separated by a space:
* the `p-value` of the statistical test;
* the number `1` if the null hypothesis is rejected at the significance level $\alpha=0.05$, or the number `0` otherwise.

The absolute error of the `p-value` must not exceed $10^{-4}$.

### Task B: Inheritance

You have been given a ready‑made binary classification pipeline. The attached baseline reports `ROC‑AUC ≈ 0.97` on its validation set.

Each row corresponds to an event involving a certain entity at the time specified by event_time. In production, the model is applied to later events. Among the future data, there are both known entities and groups, as well as those that have not been observed before.

The numerical features f_00–f_23 were obtained using the existing feature pipeline. Its documentation is incomplete, so the correctness of individual features and the original validation protocol cannot be guaranteed.

Your goal is to achieve the highest possible score on the hidden future period. You are allowed to change:
* the validation, 
* the set of features, 
* preprocessing, 
* the model, 
* the training method.

#### files:
* `train.csv             —` features and target;
* `test.csv              —` a later period without target;
* `starter.py            —` the original baseline;
* `sample_submission.csv —` a sample submission;
* `requirements.txt      —` minimal environment.

#### main columns:
* `row_id —` row identifier;
* `event_time —` event time in arbitrary seconds;
* `entity_id, group_id —` recurring categorical identifiers;
* `cat_00–cat_04 —` categorical features;
* `f_00–f_23 —` anonymous numerical features;
* `target —` binary target variable, available only in `train`.

**NB!:** Rows in `train.csv` are shuffled. Rows in `test.csv` belong to a period subsequent to `train`.

#### submission format
You must submit only a CSV file:

row_id,prediction

10000001,0.031

10000002,0.772

#### requirements:

1) exactly one row for each `row_id` from `test.csv`;
2) columns must be named `row_id` and `prediction`;
3) prediction must be a finite number between `0` and `1`;
4) extra, missing, or duplicate `row_id` values are not allowed;
5) no text answers or code need to be submitted.

#### metrics
The hidden test is divided into three consecutive time blocks. Let `A1`, `A2`, `A3` denote the `ROC‑AUC` on these blocks.

A cold‑start subset is defined separately: rows where `entity_id` or `group_id` did not appear in train. Let `Acold` denote the `ROC‑AUC` on this subset.

### robust quality $R$ is calculated first:

$$
R \;=\; 0.50 \cdot mean(A_1, A_2, A_3) \;+\; 0.25 \cdot A_{cold} \;+\; 0.25 \cdot min(A_1, A_2, A_3, A_{cold})
$$

##### final score:

$$
Score \;=\; 100 \;\cdot\; clip\left(\,\frac{(R - 0.55)}{0.20}, \, 0, \, 1\right)
$$

##### benchmarks:
* `0 points  :` R ≤ 0.55;
* `50 points :` R = 0.65;
* `75 points :` R = 0.70;
* `100 points:` R ≥ 0.75.

The higher the score, the better. The best submission counts.