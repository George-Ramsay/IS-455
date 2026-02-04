# type: ignore

def univariate(df: pd.DataFrame):
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns

  df_results = pd.DataFrame(columns=["Data Type", "Count", "Missing", "Unique", "Mode", "Min", "Q1", "Median",
                                     "Q3", "Max", "Mean", "Std", "Skew", "Kurt"])

  for col in df.columns:
    df_results.loc[col, "Data Type"] = df[col].dtype
    df_results.loc[col, "Count"] = df[col].count()
    df_results.loc[col, "Missing"] = df[col].isna().sum()
    df_results.loc[col, "Unique"] = df[col].nunique()
    df_results.loc[col, "Mode"] = df[col].mode()[0]

    if df[col].dtype in ["int64", "float64"]:
      df_results.loc[col, "Min"] = df[col].min()
      df_results.loc[col, "Q1"] = df[col].quantile(0.25)
      df_results.loc[col, "Median"] = df[col].median()
      df_results.loc[col, "Q3"] = df[col].quantile(0.75)
      df_results.loc[col, "Max"] = df[col].max()
      df_results.loc[col, "Mean"] = df[col].mean()
      df_results.loc[col, "Std"] = df[col].std()
      df_results.loc[col, "Skew"] = df[col].skew()
      df_results.loc[col, "Kurt"] = df[col].kurt()

      # Check if column is NOT boolean 0/1
      unique_vals = set(df[col].dropna().unique())
      is_boolean = unique_vals.issubset({0, 1})
      
      if not is_boolean:
        # Create stacked plot: horizontal box on top, histogram with KDE underneath (shared x-axis)
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(10, 6),
                                            gridspec_kw={"height_ratios": (.15, .85)})
        sns.set_style('ticks')

        flierprops = dict(marker='o', markersize=4, markerfacecolor='none', linestyle='none', markeredgecolor='gray')
        sns.boxplot(data=df, x=col, ax=ax_box, fliersize=4, saturation=0.50, width=0.50, linewidth=0.5, flierprops=flierprops)
        sns.histplot(data=df, x=col, ax=ax_hist, kde=True, color="orange")

        ax_box.set(yticks=[])
        ax_box.set(xticks=[])
        ax_box.set_xlabel('')
        ax_box.set_ylabel('')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_xlabel(col)
        plt.suptitle(f'Box Plot and Distribution for {col}', y=1.02)
        sns.despine(ax=ax_hist)
        sns.despine(ax=ax_box, left=True, bottom=True)
        plt.tight_layout()
        plt.show()
    else:
      # Prepare for categorical plots
      plt.figure(figsize=(10, 6))
      ax = sns.countplot(data=df, x=col)
      plt.title(f'Count Plot for {col}')
      plt.xlabel(col)
      plt.ylabel('Count')
      plt.xticks(rotation=45, ha='right')
      
      # Add percentage labels above each bar
      total = len(df[col].dropna())
      for p in ax.patches:
        height = p.get_height()
        percentage = (height / total) * 100
        ax.text(p.get_x() + p.get_width() / 2., height,
                f'{percentage:.1f}%',
                ha='center', va='bottom')
      
      plt.tight_layout()
      plt.show()

  return df_results

def drop_columns(df: pd.DataFrame):
  import pandas as pd

  for col in df.columns:
    if df[col].nunique() == 1:
      df.drop(col, axis=1, inplace=True)

    # Drop any column where the number of unique values equals the number of rows
    # and if the column is not numeric
    elif df[col].nunique() == df.shape[0]:
      if df[col].dtype == 'object':
        df.drop(col, axis=1, inplace=True)

  return df

def bin_categories(df: pd.DataFrame, columns=None, min_percent=0.05, min_count=15, drop_below_threshold_other=False):
  import pandas as pd

  # If columns is None or empty list, apply to every column; otherwise only the listed columns
  cols_to_process = list(df.columns) if (columns is None or len(columns) == 0) else columns

  for col in cols_to_process:
    n_total = len(df)
    if col not in df.columns:
      continue  # skip if column name not in dataframe
    if df[col].dtype == 'object':
      value_counts = df[col].value_counts()
      # Keep a category if it meets EITHER threshold (count >= min_count OR percent >= min_percent)
      to_bin = []
      for val, count in value_counts.items():
        pct = count / n_total
        if count < min_count and pct < min_percent:
          to_bin.append(val)
      df[col] = df[col].replace(to_bin, 'Other')

      # Optionally drop rows where 'Other' doesn't meet either threshold
      if drop_below_threshold_other and 'Other' in df[col].values:
        other_count = (df[col] == 'Other').sum()
        other_pct = other_count / len(df)
        if other_count < min_count and other_pct < min_percent:
          df.drop(df[df[col] == 'Other'].index, inplace=True)

  return df

def missing_data_diagnostics(df: pd.DataFrame, missing_thresh=0.9, verbose=True):
  """
  Report missing data counts, proportions, and heuristic suggested mechanism (MCAR/MAR/MNAR).
  Does not modify the dataframe. MAR vs MNAR cannot be distinguished from data alone;
  the function reports 'MAR/MNAR' when missingness is associated with observed variables.
  """
  import pandas as pd
  import numpy as np
  from scipy import stats

  n_rows, n_cols = len(df), len(df.columns)
  if n_rows == 0 or n_cols == 0:
    if verbose:
      print("DataFrame is empty.")
    return {"summary": "empty", "cols_dropped": [], "rows_dropped": [], "per_column": {}}

  # Per-column missing counts and proportions
  missing_count = df.isna().sum()
  missing_prop = missing_count / n_rows
  cols_with_missing = missing_count[missing_count > 0].index.tolist()

  # Columns that would be dropped at missing_thresh (drop if proportion > thresh)
  cols_to_drop = missing_prop[missing_prop > missing_thresh].index.tolist()
  cols_after_drop = [c for c in df.columns if c not in cols_to_drop]
  n_cols_after = len(cols_after_drop)

  # Rows that would be dropped: use remaining columns only
  if n_cols_after > 0:
    row_missing_count = df[cols_after_drop].isna().sum(axis=1)
    row_missing_prop = row_missing_count / n_cols_after
    rows_to_drop_mask = row_missing_prop > missing_thresh
    rows_to_drop_count = rows_to_drop_mask.sum()
  else:
    rows_to_drop_count = n_rows

  # Heuristic mechanism per column with missingness
  alpha = 0.05
  per_column = {}
  for col in df.columns:
    n_miss = missing_count[col]
    if n_miss == 0:
      per_column[col] = {"missing_count": 0, "missing_prop": 0.0, "suggested_mechanism": "no missing"}
      continue
    prop = missing_prop[col]
    mechanism = "MCAR?"
    other_cols = [c for c in df.columns if c != col and df[c].notna().any()]
    for other in other_cols:
      if df[other].isna().all():
        continue
      mask_missing = df[col].isna()
      obs_missing = df.loc[mask_missing, other].dropna()
      obs_observed = df.loc[~mask_missing, other].dropna()
      if len(obs_missing) < 2 or len(obs_observed) < 2:
        continue
      if pd.api.types.is_numeric_dtype(df[other]):
        try:
          _, p = stats.ttest_ind(obs_missing, obs_observed, nan_policy="omit")
          if p is not None and not np.isnan(p) and p < alpha:
            mechanism = "MAR/MNAR"
            break
        except Exception:
          pass
      else:
        try:
          ctab = pd.crosstab(df[other].fillna("__NA__"), mask_missing.astype(int))
          if ctab.size >= 2 and ctab.shape[0] >= 1 and ctab.shape[1] == 2:
            _, p, _, _ = stats.chi2_contingency(ctab)
            if p is not None and p < alpha:
              mechanism = "MAR/MNAR"
              break
        except Exception:
          pass
    per_column[col] = {"missing_count": int(n_miss), "missing_prop": float(prop), "suggested_mechanism": mechanism}

  result = {
    "per_column": per_column,
    "cols_dropped": cols_to_drop,
    "rows_dropped_count": int(rows_to_drop_count) if n_cols_after > 0 else n_rows,
    "missing_thresh": missing_thresh,
  }

  if verbose:
    print("=== Missing data diagnostics ===")
    print(f"Threshold: drop if proportion missing > {missing_thresh}")
    print(f"Columns that would be dropped ({len(cols_to_drop)}): {cols_to_drop or 'none'}")
    print(f"Rows that would be dropped: {result['rows_dropped_count']}")
    if not cols_with_missing:
      print("No missing values in any column.")
    else:
      print("\nPer-column summary (columns with missing):")
      for col in cols_with_missing:
        info = per_column[col]
        print(f"  {col}: missing={info['missing_count']} ({info['missing_prop']:.2%}), suggested mechanism={info['suggested_mechanism']}")
    print("(MAR vs MNAR cannot be distinguished from data; 'MAR/MNAR' means missingness is associated with observed variables.)")

  return result

def missing_data_clean(df: pd.DataFrame, missing_thresh=0.9, imputation_level='simple', diagnostics=False, missing_indicator=False):
  """
  Return a cleaned pandas DataFrame with all missing values appropriately filled in.
  (1) Drops columns/rows with proportion missing > missing_thresh;
  (2) Imputes remaining missing values (numeric: median/KNN/MICE; categorical: mode).
  Optionally prints diagnostics first or adds missing-indicator columns.
  """
  import pandas as pd
  import numpy as np
  from sklearn.impute import SimpleImputer, KNNImputer
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer

  out = df.copy()
  n_rows, n_cols = len(out), len(out.columns)
  if n_rows == 0 or n_cols == 0:
    return pd.DataFrame(out)

  if diagnostics:
    missing_data_diagnostics(out, missing_thresh=missing_thresh, verbose=True)

  # Drop columns with proportion missing > missing_thresh
  missing_per_col = out.isna().sum() / n_rows
  cols_drop = missing_per_col[missing_per_col > missing_thresh].index.tolist()
  out = out.drop(columns=cols_drop, errors="ignore")
  if len(out.columns) == 0:
    return pd.DataFrame(out)

  # Drop rows with proportion missing > missing_thresh (over remaining columns)
  n_c = len(out.columns)
  row_miss = out.isna().sum(axis=1) / n_c
  out = out.loc[row_miss <= missing_thresh].copy()

  if len(out) == 0:
    return pd.DataFrame(out)

  if not out.isna().any().any():
    return pd.DataFrame(out)

  # Optional missing indicators (add before imputation)
  if missing_indicator:
    cols_with_missing = [c for c in out.columns if out[c].isna().any()]
    for col in cols_with_missing:
      ind_name = f"{col}_was_missing"
      out[ind_name] = out[col].isna().astype(int)

  numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
  other_cols = [c for c in out.columns if c not in numeric_cols]

  # Impute numeric columns (assign back into DataFrame so index/columns are preserved)
  if numeric_cols:
    X_num = out[numeric_cols]
    if imputation_level == 'simple':
      imp_num = SimpleImputer(strategy='median')
      imputed = imp_num.fit_transform(X_num)
    elif imputation_level == 'knn':
      imp_num = KNNImputer()
      imputed = imp_num.fit_transform(X_num)
    elif imputation_level == 'mice':
      imp_num = IterativeImputer(max_iter=10, random_state=0)
      imputed = imp_num.fit_transform(X_num)
    else:
      imp_num = SimpleImputer(strategy='median')
      imputed = imp_num.fit_transform(X_num)
    out[numeric_cols] = pd.DataFrame(imputed, index=out.index, columns=numeric_cols)

  # Impute categorical / object columns with mode
  for col in other_cols:
    if out[col].isna().any():
      mode_val = out[col].mode()
      fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
      out[col] = out[col].fillna(fill_val)

  # Final pass: fill any remaining missing (e.g. datetime or other dtypes)
  for col in out.columns:
    if out[col].isna().any():
      if pd.api.types.is_numeric_dtype(out[col]):
        out[col] = out[col].fillna(out[col].median())
      else:
        mode_val = out[col].mode()
        fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
        out[col] = out[col].fillna(fill_val)

  return pd.DataFrame(out)

def manage_dates(df: pd.DataFrame, columns=None, startdate=None, enddate=None, date_threshold=0.5):
  """
  Convert columns that are valid dates into datetime format and add new features:
  day, month, year, weekday, and hour (only if the column includes a time component).
  If startdate or enddate is specified, add columns with the number of days between
  that date and the date column value.
  """
  import pandas as pd

  out = df.copy()
  cols_to_check = list(columns) if columns is not None else list(out.columns)
  cols_to_check = [c for c in cols_to_check if c in out.columns]

  for col in cols_to_check:
    if pd.api.types.is_datetime64_any_dtype(out[col]):
      ser = out[col]
    elif pd.api.types.is_numeric_dtype(out[col]):
      continue
    else:
      parsed = pd.to_datetime(out[col], errors="coerce")
      non_null = out[col].notna()
      if non_null.sum() == 0:
        continue
      pct_valid = parsed.notna().loc[non_null].sum() / non_null.sum()
      if pct_valid < date_threshold:
        continue
      ser = parsed

    out[col] = ser
    out[f"{col}_day"] = ser.dt.day
    out[f"{col}_month"] = ser.dt.month
    out[f"{col}_year"] = ser.dt.year
    out[f"{col}_weekday"] = ser.dt.weekday
    has_time = (ser.dt.hour != 0) | (ser.dt.minute != 0) | (ser.dt.second != 0)
    if has_time.any():
      out[f"{col}_hour"] = ser.dt.hour
    if startdate is not None:
      start = pd.Timestamp(startdate)
      out[f"{col}_days_since_start"] = (ser - start).dt.days
    if enddate is not None:
      end = pd.Timestamp(enddate)
      out[f"{col}_days_until_end"] = (end - ser).dt.days

  return out

def normalize(df: pd.DataFrame, columns=None, skew_thresh=0.5, inplace=False, return_report=False):
  """
  Normalize skewed numeric features by applying transformations and selecting
  the one with skewness closest to 0.

  Parameters:
    df (pd.DataFrame): Input dataframe.
    columns (list|None): Columns to consider. If None, use all numeric columns.
    skew_thresh (float): Apply transformation only if |skew| > skew_thresh.
    inplace (bool): If True, modify df in place; otherwise return a copy.
    return_report (bool): If True, return (df, report) with chosen transforms.
  """
  import pandas as pd
  import numpy as np
  try:
    import scipy.stats as stats
  except Exception:
    stats = None

  out = df if inplace else df.copy()

  if columns is None:
    cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
  else:
    cols = [c for c in columns if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]

  report = {}

  def _shift_for_nonneg(s):
    min_val = s.min(skipna=True)
    if pd.isna(min_val):
      return 0.0
    return abs(min_val) + 1e-6 if min_val <= 0 else 0.0

  for col in cols:
    ser = out[col].astype(float)
    if ser.dropna().empty:
      continue

    orig_skew = float(ser.skew())
    report[col] = {"original_skew": orig_skew, "applied": None, "best_skew": orig_skew, "shift": 0.0, "lambda": None}

    if abs(orig_skew) <= skew_thresh:
      continue

    candidates = []

    if orig_skew > 0:
      shift = _shift_for_nonneg(ser)
      candidates.append(("sqrt", np.sqrt(ser + shift), {"shift": shift}))
      candidates.append(("cbrt", np.cbrt(ser), {"shift": 0.0}))
      candidates.append(("fourth_root", np.power(ser + shift, 0.25), {"shift": shift}))
      candidates.append(("log", np.log1p(ser + shift), {"shift": shift}))
    else:
      candidates.append(("square", np.power(ser, 2), {"shift": 0.0}))
      candidates.append(("cube", np.power(ser, 3), {"shift": 0.0}))

    if stats is not None:
      try:
        yj, lam = stats.yeojohnson(ser.dropna())
        yj_ser = pd.Series(yj, index=ser.dropna().index)
        yj_full = ser.copy()
        yj_full.loc[yj_ser.index] = yj_ser
        candidates.append(("yeojohnson", yj_full, {"shift": 0.0, "lambda": float(lam)}))
      except Exception:
        pass

    best_name = None
    best_skew = None
    best_series = None
    best_meta = {"shift": 0.0, "lambda": None}

    for name, transformed, meta in candidates:
      if transformed.dropna().empty:
        continue
      s = float(pd.Series(transformed).skew())
      if best_skew is None or abs(s) < abs(best_skew):
        best_skew = s
        best_name = name
        best_series = transformed
        best_meta = {"shift": float(meta.get("shift", 0.0)), "lambda": meta.get("lambda", None)}

    if best_name is not None and best_series is not None:
      out[col] = best_series
      report[col]["applied"] = best_name
      report[col]["best_skew"] = float(best_skew)
      report[col]["shift"] = best_meta.get("shift", 0.0)
      report[col]["lambda"] = best_meta.get("lambda", None)

  return (out, report) if return_report else out
