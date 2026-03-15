# March Madness ML Plan

## Objective

Build the strongest **classification** workflow for predicting NCAA Tournament game winners while keeping the process:

- reproducible
- low-mess
- easy to inspect
- easy to improve from one notebook iteration to the next

Primary score to optimize: **held-out test accuracy**  
Secondary checks: cross-validation stability, confusion matrix, and a short error analysis.

## Challenge Constraints

- The label is `WIN` (`1`) vs `LOSS` (`0`).
- Use `random_state=27` anywhere a random state is accepted.
- Use only **pre-game** information.
- Do **not** use leakage features such as tournament scores, opponent scores, current/earned round information, or any other post-game outcomes.
- Final deliverable is a runnable notebook with clearly printed evaluation metrics.

## Textbook-Guided Approach

This plan follows the course workflow, adapted from regression examples to a classification problem:

- **Chapter 1:** use an iterative CRISP-DM style process instead of a one-shot notebook.
- **Chapter 6:** automate repeatable EDA so we inspect data quickly and consistently.
- **Chapter 7:** build reusable data-prep steps for missingness, categories, dates, and skew.
- **Chapter 11:** focus on prediction, leakage prevention, train/test discipline, and pipelines that learn from training data only.
- **Chapter 12:** compare interpretable tree-based models and tune them with regularization instead of guessing.

Practical adaptation: where the textbook uses predictive regression, we will use the classification equivalent:

- baseline linear model -> `LogisticRegression`
- regression tree ideas -> classification trees / tree ensembles
- MAE/RMSE emphasis -> accuracy-first evaluation for this challenge

## Working Principles

- Keep one clear notebook per iteration.
- Change a small number of things per iteration so gains are explainable.
- Freeze the validation strategy early so model comparisons stay fair.
- Reuse code we trust; only promote new helper code after it proves useful in more than one iteration.
- Record what changed, what improved, and what still looks weak at the end of every notebook.

## Reusable Code From `ML-Pipeline-Kit/ml_library.py`

Use confidently:

- `univariate(df)` for fast feature-level EDA
- `drop_columns(df)` to remove constant or obvious ID-like object columns
- `bin_categories(...)` for rare-category consolidation
- `missing_data_diagnostics(...)` and `missing_data_clean(...)` for a documented missing-data pass
- `manage_dates(...)` when date columns appear in the challenge files
- `normalize(...)` for skew reduction when testing linear models

Use selectively:

- `manage_vif(...)` only for interpretable linear/logistic baselines
- `manage_outliers(...)` only as an experiment, not as a default

Reason: in March Madness data, extreme teams and upset patterns may be real signal, not bad data.

## Core Modeling Design

We should build the modeling table from the perspective of a **team facing an opponent**:

1. Start from `Tournament Matchups.csv`.
2. Derive the target (`WIN`/`LOSS`) from the game result.
3. Pair each team with its opponent using the assignment's matchup logic and any game keys available in the file.
4. Merge in only pre-game team information from allowed source tables.
5. Create matchup features as **differences** or contrasts between team and opponent values.

Examples of likely useful matchup features:

- seed difference
- offensive efficiency difference
- defensive efficiency difference
- SOS difference
- resume / WAB / ELO difference
- shooting split differences
- rebound / turnover / pace differences
- travel-distance and time-zone differences
- coach / conference / tournament experience indicators

This keeps the table interpretable and lets us inspect which gaps matter most.

## Validation Strategy

Use the same evaluation structure in every iteration:

- Hold out a final test set once and keep it fixed.
- Prefer **year-aware** validation if possible so seasons do not mix too freely.
- If grouping by `YEAR` is practical, use grouped CV for model selection and keep the holdout untouched until the end of each notebook.
- Report:
  - train accuracy
  - validation / CV accuracy
  - held-out test accuracy
  - confusion matrix
  - short notes on major false positives / false negatives

If grouped validation is not workable because of data shape, fall back to a stratified split, but document why.

## Notebook Structure Template

Each notebook should follow the same structure:

1. Goal of this iteration
2. Imports, paths, and `random_state=27`
3. Data loading
4. Label creation and leakage audit
5. EDA summary
6. Feature engineering / preprocessing
7. Modeling
8. Evaluation
9. Error analysis
10. Decision for next iteration

Each notebook should end with a short table like this:

| Iteration | What Changed | Best Validation Score | Test Score | Keep / Drop Next Time |
| --- | --- | --- | --- | --- |

## Iteration Roadmap

### Iteration 1: Data Audit + Baseline

Goal: create a clean, trustworthy first benchmark.

Deliverables:

- load the matchup file and inspect schema
- derive the binary target correctly
- identify and remove obvious leakage columns
- build the first modeling table with only the simplest safe features
- run automated EDA with `univariate`
- create a baseline comparison:
  - `DummyClassifier`
  - `LogisticRegression`
  - a shallow `DecisionTreeClassifier`

Questions to answer:

- Is the label balanced as expected?
- Which columns are clearly safe vs clearly leaked?
- What is the baseline accuracy from seeds and a very small feature set?
- What data quality issues appear immediately?

Exit condition:

- one reproducible baseline notebook
- one frozen split strategy
- one list of safe starter features

### Iteration 2: Core Feature Assembly

Goal: add the strongest pre-game team strength information.

Priority source tables from the challenge brief:

- `KenPom Barttorvik.csv`
- `TeamRankings.csv`
- `Resumes.csv`
- `Tournament Locations.csv`

Deliverables:

- standardize merge keys across files
- merge the highest-value tables first
- engineer matchup-difference features
- keep a short data dictionary for new features
- compare regularized logistic regression against a simple tree-based model

Questions to answer:

- Which external tables add signal fastest?
- Are raw team stats or team-opponent differences more predictive?
- Which features consistently appear among top coefficients / importances?

Exit condition:

- a stronger feature table than Iteration 1
- a short ranked list of useful feature groups

### Iteration 3: Data Preparation Refinement

Goal: clean the feature table without turning the workflow into a black box.

Deliverables:

- use `drop_columns` for low-value columns
- run `missing_data_diagnostics` and choose a justified imputation approach
- apply `bin_categories` where rare categories are noisy
- use `manage_dates` if date fields matter
- test `normalize` for linear models only
- keep preprocessing inside a scikit-learn pipeline where fitting is learned from training data only

Questions to answer:

- Which missing-data strategy is most stable?
- Do rare-category bins help or hide useful signal?
- Do transformed numeric features help logistic regression materially?

Exit condition:

- one preprocessing recipe we trust
- fewer manual notebook-only cleaning steps

### Iteration 4: Model Sweep and Tuning

Goal: test whether nonlinear models beat the simpler baseline enough to justify their complexity.

Priority models:

- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `XGBoost`, `LightGBM`, or `CatBoost` if available and worth the setup

Tuning focus:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- learning rate / number of estimators for boosting models

Questions to answer:

- Does nonlinear structure materially improve test accuracy?
- Which model gives the best balance of performance and explainability?
- Is any model overfitting based on train vs validation gaps?

Exit condition:

- one strongest simple model
- one strongest high-performance model

### Iteration 5: Final Tightening

Goal: make the best model presentation-ready and defensible.

Deliverables:

- remove feature clutter that is not helping
- rerun the final comparison on the fixed split
- inspect the final error cases
- print clean final metrics
- summarize why the winning model won

Optional only if it clearly helps:

- soft voting / ensembling
- probability calibration
- threshold experimentation

Do not add these if they make the workflow harder to explain for only a tiny gain.

Exit condition:

- final notebook is concise, reproducible, and easy to present

## What We Will Avoid

- giant one-notebook experimentation
- changing split logic every iteration
- using every CSV before proving value
- aggressive outlier deletion without evidence
- heavy feature engineering that cannot be explained
- leakage hidden inside preprocessing done before splitting

## Lightweight Experiment Tracking

At the end of each iteration, record:

- feature sources used
- preprocessing choices
- model(s) tested
- validation metric
- held-out test metric
- top error pattern noticed
- exact next change to try

This keeps the next notebook focused instead of exploratory in every direction.

## Immediate Next Step

Build `iter_1.ipynb` as the baseline notebook:

- load the challenge files
- derive the target
- perform the first leakage audit
- establish the fixed split strategy
- benchmark a dummy model, logistic regression, and a shallow decision tree
