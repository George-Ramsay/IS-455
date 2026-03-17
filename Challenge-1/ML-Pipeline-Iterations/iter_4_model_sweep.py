from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 27
MAX_BACKWARD_ELIMINATION_FEATURES = 32
ELIMINATION_PROGRESS_EVERY = 10
ITER_2_FINAL_FEATURES = [
    "seed_diff",
    "diff_kp_kadj_em",
    "diff_kp_barthag",
    "diff_tr_sos_rating",
    "diff_kp_badj_em_rank",
    "diff_res_b_power",
    "diff_kp_kadj_em_rank",
    "diff_tr_lo",
    "diff_tr_sos_last",
    "diff_res_r_score",
    "diff_res_wab_rank",
    "diff_kp_kadj_o",
]

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed`", category=UserWarning)


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "AGENTS.md").exists() and (candidate / "Challenge-1").exists():
            return candidate
    raise FileNotFoundError("Could not find the IS-455 project root from the current notebook location.")


PROJECT_ROOT = find_project_root(Path.cwd())
CHALLENGE_DIR = PROJECT_ROOT / "Challenge-1"
KAGGLE_DATA_DIR = CHALLENGE_DIR / "data-Kaggle"
BARTTORVIK_DATA_DIR = CHALLENGE_DIR / "data-BartTorvik"
ML_KIT_DIR = PROJECT_ROOT / "ML-Pipeline-Kit"

for path in [CHALLENGE_DIR, ML_KIT_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

import challenge_data_prep as prep
import ml_library as ml


def choose_cv(y_train: pd.Series, groups_train: pd.Series | None = None):
    if groups_train is not None and pd.Series(groups_train).nunique(dropna=True) >= 3:
        n_splits = min(5, pd.Series(groups_train).nunique(dropna=True))
        return GroupKFold(n_splits=n_splits), f"GroupKFold({n_splits}) by year"

    min_class_count = int(pd.Series(y_train).value_counts().min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("Not enough observations per class for cross-validation.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), f"StratifiedKFold({n_splits})"


def make_pipeline(feature_cols: list[str], estimator, scale_numeric: bool = False) -> Pipeline:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[("numeric", Pipeline(steps=numeric_steps), feature_cols)],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def make_three_way_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None) -> tuple[dict[str, np.ndarray], str]:
    index = np.arange(len(X))

    if groups is not None and pd.Series(groups).nunique(dropna=True) >= 5:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
        train_val_idx, test_idx = next(outer.split(X, y, groups=groups))

        remaining_groups = pd.Series(groups).iloc[train_val_idx]
        if remaining_groups.nunique(dropna=True) >= 4:
            inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
            rel_train_idx, rel_val_idx = next(
                inner.split(X.iloc[train_val_idx], y.iloc[train_val_idx], groups=remaining_groups)
            )
            train_idx = train_val_idx[rel_train_idx]
            val_idx = train_val_idx[rel_val_idx]
            return {
                "train": np.asarray(train_idx),
                "validation": np.asarray(val_idx),
                "test": np.asarray(test_idx),
            }, "Grouped 60/20/20 split by year via GroupShuffleSplit"

    train_val_idx, test_idx = train_test_split(index, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        stratify=y.iloc[train_val_idx],
        random_state=RANDOM_STATE,
    )
    return {
        "train": np.asarray(train_idx),
        "validation": np.asarray(val_idx),
        "test": np.asarray(test_idx),
    }, "Stratified 60/20/20 split via train_test_split"


def select_candidate_features(df: pd.DataFrame, blocked_columns: list[str], baseline_features: list[str]) -> list[str]:
    blocked = set(blocked_columns)
    blocked.update(f"opp_{column}" for column in blocked_columns if not column.startswith("opp_"))

    exact_exclude = {
        "win",
        "year",
        "by_year_no",
        "current_round",
        "game_index",
        "game_id",
        "slot_in_game",
        "team_no",
        "opp_team_no",
        "seed",
        "opp_seed",
    }
    exact_exclude.update(blocked)

    numeric_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    selected: list[str] = []
    for column in numeric_cols:
        if column in exact_exclude or column.endswith("_id"):
            continue
        if column.startswith("opp_") and f"diff_{column[4:]}" in df.columns:
            continue
        if not column.startswith("diff_") and f"diff_{column}" in df.columns:
            continue
        selected.append(column)

    ordered: list[str] = []
    for column in baseline_features:
        if column in selected and column not in ordered:
            ordered.append(column)
    for column in sorted(selected):
        if column not in ordered:
            ordered.append(column)
    return ordered


def rank_candidate_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: list[str],
    baseline_features: list[str],
) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for column in candidate_features:
        series = X_train[column]
        corr = series.corr(y_train)
        rows.append(
            {
                "feature": column,
                "abs_train_corr": float(abs(corr)) if pd.notna(corr) else -1.0,
                "missing_pct": float(series.isna().mean()),
            }
        )

    ranking = pd.DataFrame(rows).sort_values(
        ["abs_train_corr", "missing_pct", "feature"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    ranked: list[str] = [column for column in baseline_features if column in candidate_features]
    for column in ranking["feature"]:
        if column not in ranked:
            ranked.append(column)

    ranking["rank_order"] = ranking["feature"].map({feature: idx + 1 for idx, feature in enumerate(ranked)})
    return ranked, ranking


def cap_features_for_backward_elimination(
    ranked_features: list[str],
    baseline_features: list[str],
    max_features: int,
) -> list[str]:
    if len(ranked_features) <= max_features:
        return list(ranked_features)

    locked = [feature for feature in baseline_features if feature in ranked_features]
    capped = list(locked)
    for feature in ranked_features:
        if feature not in capped:
            capped.append(feature)
        if len(capped) >= max_features:
            break
    return capped


def cross_validated_scores(
    feature_cols: list[str],
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series | None = None,
    scale_numeric: bool = False,
) -> tuple[np.ndarray, str]:
    pipeline = make_pipeline(feature_cols, estimator, scale_numeric=scale_numeric)
    cv, cv_name = choose_cv(y_train=y_train, groups_train=groups_train)
    if isinstance(cv, GroupKFold):
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, groups=groups_train, scoring="accuracy")
    else:
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    return scores, cv_name


def evaluate_candidate(
    label: str,
    family: str,
    feature_set_name: str,
    feature_cols: list[str],
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    groups_train: pd.Series | None = None,
    scale_numeric: bool = False,
) -> tuple[dict[str, Any], Pipeline]:
    cv_scores, cv_name = cross_validated_scores(
        feature_cols=feature_cols,
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        scale_numeric=scale_numeric,
    )
    pipeline = make_pipeline(feature_cols, estimator, scale_numeric=scale_numeric)
    pipeline.fit(X_train, y_train)
    train_pred = pipeline.predict(X_train)
    eval_pred = pipeline.predict(X_eval)

    train_accuracy = accuracy_score(y_train, train_pred)
    validation_accuracy = accuracy_score(y_eval, eval_pred)
    summary = {
        "label": label,
        "family": family,
        "feature_set": feature_set_name,
        "feature_count": len(feature_cols),
        "cv_strategy": cv_name,
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_median": float(np.median(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "train_accuracy": float(train_accuracy),
        "validation_accuracy": float(validation_accuracy),
        "train_val_gap": float(train_accuracy - validation_accuracy),
    }
    return summary, pipeline


def leaderboard_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["cv_accuracy_median", "cv_accuracy_mean", "validation_accuracy", "train_val_gap", "label"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def run_backward_elimination(
    feature_cols: list[str],
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series | None = None,
    tolerance: float = 0.001,
    min_features: int = 3,
    scale_numeric: bool = False,
) -> tuple[list[str], pd.DataFrame, str]:
    current_features = list(feature_cols)
    current_scores, cv_name = cross_validated_scores(
        feature_cols=current_features,
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        scale_numeric=scale_numeric,
    )
    current_mean = float(np.mean(current_scores))
    current_median = float(np.median(current_scores))
    current_std = float(np.std(current_scores))

    history = [
        {
            "step": 0,
            "removed_feature": "<none>",
            "remaining_feature_count": len(current_features),
            "cv_accuracy_mean": current_mean,
            "cv_accuracy_median": current_median,
            "cv_accuracy_std": current_std,
            "accepted": True,
        }
    ]

    step = 0
    while len(current_features) > min_features:
        print(
            f"Backward elimination step {step + 1}: testing {len(current_features)} removal candidates "
            f"with {len(current_features)} remaining features"
        )
        trials = []
        for feature_idx, feature in enumerate(current_features, start=1):
            trial_features = [column for column in current_features if column != feature]
            trial_scores, _ = cross_validated_scores(
                feature_cols=trial_features,
                estimator=estimator,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                scale_numeric=scale_numeric,
            )
            trials.append(
                {
                    "removed_feature": feature,
                    "trial_features": trial_features,
                    "cv_accuracy_mean": float(np.mean(trial_scores)),
                    "cv_accuracy_median": float(np.median(trial_scores)),
                    "cv_accuracy_std": float(np.std(trial_scores)),
                }
            )
            if feature_idx % ELIMINATION_PROGRESS_EVERY == 0 or feature_idx == len(current_features):
                print(
                    f"  evaluated {feature_idx}/{len(current_features)} candidate removals "
                    f"for step {step + 1}"
                )

        trial_frame = pd.DataFrame(trials).sort_values(
            ["cv_accuracy_median", "cv_accuracy_mean", "cv_accuracy_std", "removed_feature"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        best_trial = trial_frame.iloc[0]
        best_trial_median = float(best_trial["cv_accuracy_median"])
        if best_trial_median + tolerance < current_median:
            print(
                f"Stopping elimination at step {step + 1}: best median CV {best_trial_median:.3f} "
                f"would drop below current {current_median:.3f} beyond tolerance {tolerance:.3f}"
            )
            break

        step += 1
        current_features = list(best_trial["trial_features"])
        current_mean = float(best_trial["cv_accuracy_mean"])
        current_median = float(best_trial["cv_accuracy_median"])
        current_std = float(best_trial["cv_accuracy_std"])
        print(
            f"Accepted removal: {best_trial['removed_feature']} | remaining features: {len(current_features)} | "
            f"median CV: {current_median:.3f}"
        )
        history.append(
            {
                "step": step,
                "removed_feature": str(best_trial["removed_feature"]),
                "remaining_feature_count": len(current_features),
                "cv_accuracy_mean": current_mean,
                "cv_accuracy_median": current_median,
                "cv_accuracy_std": current_std,
                "accepted": True,
            }
        )

    return current_features, pd.DataFrame(history), cv_name


def holdout_audit(
    label: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    paired_games: pd.DataFrame,
    test_idx: np.ndarray,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"{label} test accuracy: {test_accuracy:.3f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, test_pred, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{label} confusion matrix")
    plt.show()

    print(classification_report(y_test, test_pred, digits=3))

    test_view = paired_games.iloc[test_idx].copy().reset_index(drop=True)
    test_view["actual"] = y_test.reset_index(drop=True)
    test_view["predicted"] = pd.Series(test_pred)
    if hasattr(pipeline, "predict_proba"):
        test_view["pred_win"] = pipeline.predict_proba(X_test)[:, 1]

    error_cols = [
        column
        for column in ["year", "round", "game_id", "team", "opp_team", "seed_diff", "actual", "predicted", "pred_win"]
        if column in test_view.columns
    ]
    errors = (
        test_view.loc[test_view["actual"] != test_view["predicted"], error_cols]
        .sort_values("pred_win", ascending=False, na_position="last")
        .head(20)
        .reset_index(drop=True)
    )
    display(errors)

    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = (
            pd.Series(model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
            .head(15)
            .rename("importance")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        display(importances)
    else:
        importances = pd.DataFrame(columns=["feature", "importance"])

    return errors, importances


def run_iter_4() -> dict[str, Any]:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Challenge directory: {CHALLENGE_DIR}")
    print(f"Kaggle data directory: {KAGGLE_DATA_DIR}")
    print(f"BartTorvik data directory: {BARTTORVIK_DATA_DIR}")

    bundle = prep.build_modeling_table(
        kaggle_data_dir=KAGGLE_DATA_DIR,
        barttorvik_data_dir=BARTTORVIK_DATA_DIR,
        include_optional_sources=True,
        include_auto_diffs=True,
    )
    paired_games = bundle.modeling_table.copy()

    print("Prepared table shapes:")
    print(
        json.dumps(
            {
                "team_features": bundle.merge_report["team_features_shape"],
                "matchup_rows": bundle.merge_report["matchup_rows_shape"],
                "modeling_table": bundle.merge_report["modeling_table_shape"],
            },
            indent=2,
        )
    )

    print("Selected sources:")
    display(pd.DataFrame(bundle.merge_report["team_feature_sources"]))

    print("Location merge report:")
    display(pd.DataFrame([bundle.merge_report["location_merge"]]))

    leakage_cols = bundle.merge_report["blocked_feature_columns"]
    print("Blocked feature columns:")
    display(pd.DataFrame({"blocked_feature": leakage_cols}))

    label_balance = paired_games["win"].value_counts(normalize=True).rename("share").sort_index()
    print("Label balance:")
    display(label_balance.to_frame())

    X = paired_games.copy()
    y = paired_games["win"].copy()
    groups = paired_games["year"].copy() if "year" in paired_games.columns else None

    split_indices, split_name = make_three_way_split(X=X, y=y, groups=groups)
    train_idx = split_indices["train"]
    val_idx = split_indices["validation"]
    test_idx = split_indices["test"]

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()
    y_test = y.iloc[test_idx].copy()
    groups_train = groups.iloc[train_idx].copy() if isinstance(groups, pd.Series) else None

    split_summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(train_idx), "share": len(train_idx) / len(X)},
            {"split": "validation", "rows": len(val_idx), "share": len(val_idx) / len(X)},
            {"split": "test", "rows": len(test_idx), "share": len(test_idx) / len(X)},
        ]
    )
    print(f"Split strategy: {split_name}")
    display(split_summary)

    safe_candidate_features = select_candidate_features(
        df=paired_games,
        blocked_columns=leakage_cols,
        baseline_features=ITER_2_FINAL_FEATURES,
    )
    ranked_candidate_features, feature_ranking = rank_candidate_features(
        X_train=X_train,
        y_train=y_train,
        candidate_features=safe_candidate_features,
        baseline_features=ITER_2_FINAL_FEATURES,
    )
    elimination_candidate_features = cap_features_for_backward_elimination(
        ranked_features=ranked_candidate_features,
        baseline_features=ITER_2_FINAL_FEATURES,
        max_features=MAX_BACKWARD_ELIMINATION_FEATURES,
    )
    selected_features, elimination_history, elimination_cv_name = run_backward_elimination(
        feature_cols=elimination_candidate_features,
        estimator=LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        tolerance=0.001,
        min_features=max(3, len(ITER_2_FINAL_FEATURES)),
        scale_numeric=True,
    )

    feature_sets = {
        "iter_2_locked": [column for column in ITER_2_FINAL_FEATURES if column in paired_games.columns],
        "all_ranked_candidates": ranked_candidate_features,
        "backward_eliminated": selected_features,
    }

    print(f"Safe numeric candidate count before elimination: {len(safe_candidate_features)}")
    print(f"Ranked candidate feature count: {len(ranked_candidate_features)}")
    print(
        f"Backward elimination candidate count after cap: {len(elimination_candidate_features)} "
        f"(max {MAX_BACKWARD_ELIMINATION_FEATURES})"
    )
    display(pd.DataFrame({"feature": ranked_candidate_features}))
    print("Backward elimination candidate subset:")
    display(pd.DataFrame({"feature": elimination_candidate_features}))

    print("Top training-ranked candidates:")
    display(feature_ranking.head(30))

    print(f"Backward elimination CV strategy: {elimination_cv_name}")
    print(f"Selected feature count: {len(selected_features)}")
    display(pd.DataFrame({"feature": selected_features}))
    print("Backward elimination history:")
    display(
        elimination_history.round(
            {
                "cv_accuracy_mean": 3,
                "cv_accuracy_median": 3,
                "cv_accuracy_std": 3,
            }
        )
    )

    display(ml.univariate(X_train[feature_sets["backward_eliminated"]]))

    baseline_summary, baseline_pipeline = evaluate_candidate(
        label="logistic_iter_2_locked",
        family="logistic_baseline",
        feature_set_name="iter_2_locked",
        feature_cols=feature_sets["iter_2_locked"],
        estimator=LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        X_train=X_train,
        y_train=y_train,
        X_eval=X_val,
        y_eval=y_val,
        groups_train=groups_train,
        scale_numeric=True,
    )
    baseline_cv_median = baseline_summary["cv_accuracy_median"]

    print(f"Baseline median CV accuracy: {baseline_cv_median:.3f}")
    display(
        pd.DataFrame([baseline_summary]).round(
            {
                "cv_accuracy_mean": 3,
                "cv_accuracy_median": 3,
                "cv_accuracy_std": 3,
                "train_accuracy": 3,
                "validation_accuracy": 3,
                "train_val_gap": 3,
            }
        )
    )

    tuning_rows: list[dict[str, Any]] = []
    tuned_pipelines: dict[str, Pipeline] = {baseline_summary["label"]: baseline_pipeline}

    decision_tree_grid = list(itertools.product([3, 5, 7, None], [2, 10], [1, 5, 10]))
    random_forest_max_features: list[Literal["sqrt"] | float] = ["sqrt", 0.5]
    random_forest_grid = list(itertools.product([None, 10], [1, 3, 5], random_forest_max_features))
    gradient_boosting_grid = list(itertools.product([100, 200], [0.05, 0.1], [2, 3], [1, 5]))

    optional_library_status = {"xgboost": False, "lightgbm": False, "catboost": False}
    for package_name in optional_library_status:
        try:
            __import__(package_name)
            optional_library_status[package_name] = True
        except ModuleNotFoundError:
            optional_library_status[package_name] = False

    print("Optional booster availability:")
    display(pd.DataFrame([optional_library_status]))

    candidate_specs: list[tuple[str, str, str, list[str], Any, bool]] = []
    logistic_backward_summary, logistic_backward_pipeline = evaluate_candidate(
        label="logistic_backward_eliminated",
        family="logistic_baseline",
        feature_set_name="backward_eliminated",
        feature_cols=feature_sets["backward_eliminated"],
        estimator=LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        X_train=X_train,
        y_train=y_train,
        X_eval=X_val,
        y_eval=y_val,
        groups_train=groups_train,
        scale_numeric=True,
    )
    logistic_backward_summary["delta_vs_baseline_cv_median"] = (
        logistic_backward_summary["cv_accuracy_median"] - baseline_cv_median
    )
    tuning_rows.append(logistic_backward_summary)
    tuned_pipelines[logistic_backward_summary["label"]] = logistic_backward_pipeline

    for feature_set_name in ["iter_2_locked", "backward_eliminated"]:
        feature_cols = feature_sets[feature_set_name]
        for max_depth, min_samples_split, min_samples_leaf in decision_tree_grid:
            candidate_specs.append(
                (
                    f"decision_tree__{feature_set_name}__depth_{max_depth}_split_{min_samples_split}_leaf_{min_samples_leaf}",
                    "decision_tree",
                    feature_set_name,
                    feature_cols,
                    DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=RANDOM_STATE,
                    ),
                    False,
                )
            )

    for max_depth, min_samples_leaf, max_features in random_forest_grid:
        candidate_specs.append(
            (
                f"random_forest__backward_eliminated__depth_{max_depth}_leaf_{min_samples_leaf}_maxfeat_{max_features}",
                "random_forest",
                "backward_eliminated",
                feature_sets["backward_eliminated"],
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                ),
                False,
            )
        )

    for n_estimators, learning_rate, max_depth, min_samples_leaf in gradient_boosting_grid:
        candidate_specs.append(
            (
                f"gradient_boosting__backward_eliminated__estimators_{n_estimators}_lr_{learning_rate}_depth_{max_depth}_leaf_{min_samples_leaf}",
                "gradient_boosting",
                "backward_eliminated",
                feature_sets["backward_eliminated"],
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    subsample=1.0,
                    random_state=RANDOM_STATE,
                ),
                False,
            )
        )

    print(f"Candidate configurations to evaluate: {len(candidate_specs)}")
    for idx, (label, family, feature_set_name, feature_cols, estimator, scale_numeric) in enumerate(candidate_specs, start=1):
        summary, pipeline = evaluate_candidate(
            label=label,
            family=family,
            feature_set_name=feature_set_name,
            feature_cols=feature_cols,
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_val,
            y_eval=y_val,
            groups_train=groups_train,
            scale_numeric=scale_numeric,
        )
        summary["delta_vs_baseline_cv_median"] = summary["cv_accuracy_median"] - baseline_cv_median
        tuning_rows.append(summary)
        tuned_pipelines[label] = pipeline
        if idx % 10 == 0:
            print(f"Finished {idx}/{len(candidate_specs)} candidates")

    tuning_results = pd.DataFrame([baseline_summary, *tuning_rows])
    tuning_results["delta_vs_baseline_cv_median"] = tuning_results["cv_accuracy_median"] - baseline_cv_median

    leaderboard = leaderboard_sort(tuning_results)

    print("Top CV-ranked results:")
    display(
        leaderboard.head(15).round(
            {
                "cv_accuracy_mean": 3,
                "cv_accuracy_median": 3,
                "cv_accuracy_std": 3,
                "train_accuracy": 3,
                "validation_accuracy": 3,
                "train_val_gap": 3,
                "delta_vs_baseline_cv_median": 3,
            }
        )
    )

    family_winners = leaderboard.groupby("family", as_index=False).first()
    family_winners = leaderboard_sort(family_winners)

    print("Best configuration by family:")
    display(
        family_winners.round(
            {
                "cv_accuracy_mean": 3,
                "cv_accuracy_median": 3,
                "cv_accuracy_std": 3,
                "train_accuracy": 3,
                "validation_accuracy": 3,
                "train_val_gap": 3,
                "delta_vs_baseline_cv_median": 3,
            }
        )
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = family_winners[["family", "validation_accuracy", "cv_accuracy_median", "delta_vs_baseline_cv_median"]].copy()
    plot_df = plot_df.melt(id_vars="family", var_name="metric", value_name="value")
    sns.barplot(data=plot_df, x="family", y="value", hue="metric", ax=ax)
    ax.axhline(baseline_cv_median, color="black", linestyle="--", linewidth=1, label="baseline median CV")
    ax.set_title("Family winners vs baseline median CV")
    ax.set_ylabel("accuracy")
    plt.show()

    selected_winner_row = leaderboard.iloc[0]
    best_simple_row = family_winners.loc[family_winners["family"] == "decision_tree"].iloc[0]
    best_high_perf_row = family_winners.loc[
        family_winners["family"].isin(["random_forest", "gradient_boosting"])
    ].iloc[0]
    baseline_row = family_winners.loc[family_winners["family"] == "logistic_baseline"].iloc[0]
    baseline_test_accuracy = accuracy_score(y_test, baseline_pipeline.predict(X_test))

    selected_rows = pd.DataFrame([selected_winner_row, baseline_row, best_simple_row, best_high_perf_row]).drop_duplicates(
        subset=["label"]
    ).reset_index(drop=True)
    selected_feature_sets = {row["label"]: feature_sets[row["feature_set"]] for _, row in selected_rows.iterrows()}

    selected_holdout_rows = []
    for _, row in selected_rows.iterrows():
        label = row["label"]
        pipeline = tuned_pipelines[label]
        test_accuracy = accuracy_score(y_test, pipeline.predict(X_test))
        selected_holdout_rows.append(
            {
                "label": label,
                "family": row["family"],
                "feature_set": row["feature_set"],
                "feature_count": row["feature_count"],
                "baseline_cv_median": baseline_cv_median,
                "cv_accuracy_mean": row["cv_accuracy_mean"],
                "cv_accuracy_median": row["cv_accuracy_median"],
                "validation_accuracy": row["validation_accuracy"],
                "test_accuracy": test_accuracy,
                "test_minus_baseline_test": test_accuracy - baseline_test_accuracy,
            }
        )

    holdout_summary = pd.DataFrame(selected_holdout_rows)
    holdout_summary["selected_by_cv"] = holdout_summary["label"] == str(selected_winner_row["label"])
    holdout_summary = holdout_summary.sort_values(
        ["selected_by_cv", "cv_accuracy_median", "cv_accuracy_mean", "validation_accuracy", "label"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    print("Held-out test comparison for selected models:")
    display(
        holdout_summary.round(
            {
                "baseline_cv_median": 3,
                "cv_accuracy_mean": 3,
                "cv_accuracy_median": 3,
                "validation_accuracy": 3,
                "test_accuracy": 3,
                "test_minus_baseline_test": 3,
            }
        )
    )

    print("Baseline logistic audit:")
    baseline_errors, baseline_importances = holdout_audit(
        label=str(baseline_row["label"]),
        pipeline=tuned_pipelines[str(baseline_row["label"])],
        X_test=X_test,
        y_test=y_test,
        paired_games=paired_games,
        test_idx=test_idx,
        feature_cols=selected_feature_sets[str(baseline_row["label"])],
    )

    print("Best simple model audit:")
    best_simple_errors, best_simple_importances = holdout_audit(
        label=str(best_simple_row["label"]),
        pipeline=tuned_pipelines[str(best_simple_row["label"])],
        X_test=X_test,
        y_test=y_test,
        paired_games=paired_games,
        test_idx=test_idx,
        feature_cols=selected_feature_sets[str(best_simple_row["label"])],
    )

    print("Best high-performance model audit:")
    best_high_perf_errors, best_high_perf_importances = holdout_audit(
        label=str(best_high_perf_row["label"]),
        pipeline=tuned_pipelines[str(best_high_perf_row["label"])],
        X_test=X_test,
        y_test=y_test,
        paired_games=paired_games,
        test_idx=test_idx,
        feature_cols=selected_feature_sets[str(best_high_perf_row["label"])],
    )

    winning_row = holdout_summary.loc[holdout_summary["selected_by_cv"]].iloc[0]
    iteration_summary = pd.DataFrame(
        [
            {
                "Iteration": "iter_4",
                "What Changed": "Started from the full safe feature set, used greedy backward elimination to choose the logistic feature subset automatically, and compared tree models against the iter_2 locked reference with CV-first ranking.",
                "Best Validation Score": winning_row["validation_accuracy"],
                "Baseline Median CV": baseline_cv_median,
                "Test Score": winning_row["test_accuracy"],
                "Winning Model": winning_row["label"],
                "Keep / Drop Next Time": "Keep the backward-elimination selector and CV-first comparison; next tighten feature clutter and summarize the recurring upset misses.",
            }
        ]
    )
    display(
        iteration_summary.round(
            {
                "Best Validation Score": 3,
                "Baseline Median CV": 3,
                "Test Score": 3,
            }
        )
    )

    return {
        "bundle": bundle,
        "paired_games": paired_games,
        "split_summary": split_summary,
        "feature_ranking": feature_ranking,
        "ranked_feature_pool": ranked_candidate_features,
        "selected_features": selected_features,
        "elimination_history": elimination_history,
        "baseline_summary": baseline_summary,
        "baseline_cv_median": baseline_cv_median,
        "leaderboard": leaderboard,
        "family_winners": family_winners,
        "holdout_summary": holdout_summary,
        "iteration_summary": iteration_summary,
        "baseline_errors": baseline_errors,
        "best_simple_errors": best_simple_errors,
        "best_high_perf_errors": best_high_perf_errors,
        "baseline_importances": baseline_importances,
        "best_simple_importances": best_simple_importances,
        "best_high_perf_importances": best_high_perf_importances,
    }


if __name__ == "__main__":
    run_iter_4()
