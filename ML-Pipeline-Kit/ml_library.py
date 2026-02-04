def univariate(df):
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

def drop_columns(df):
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

def bin_categories(df, columns=None, min_percent=0.05, min_count=15, drop_below_threshold_other=False):
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

def missing_data_diagnostics(df, missing_thresh=0.9, verbose=True):
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

def missing_data_clean(df, missing_thresh=0.9, imputation_level='simple', diagnostics=False, missing_indicator=False):
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

def manage_dates(df, columns=None, startdate=None, enddate=None, date_threshold=0.5):
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


def normalize(df, columns=None, verbose=True, keep_originals=True):
  """
  Reduce skewness of numeric features by trying a menu of transforms and keeping
  the one that brings skewness closest to zero. New values are stored in a
  column with suffix '_normalized'.

  For positive skew: tries yeojohnson, sqrt, cbrt, ln.
  For negative skew: tries yeojohnson, square, cube, exponent (x**4).

  Parameters
  ----------
  df : pandas.DataFrame
      Input dataframe.
  columns : list of str or None, optional
      Numeric columns to normalize. If None, all numeric columns are used.
  verbose : bool, default True
      If True, print a report of original skew, chosen transform, and new skew per column.
  keep_originals : bool, default True
      If True, keep original columns and add *_normalized columns.
      If False, drop original columns and rename *_normalized to the original name.

  Returns
  -------
  tuple of (pandas.DataFrame, dict)
      - First element: DataFrame with normalized column(s) added (and optionally originals removed).
      - Second element: dict mapping each output column name to the exact transformation applied.
        For Yeo-Johnson the value is the lambda (float); otherwise the transform name
        ('sqrt', 'cbrt', 'ln', 'square', 'cube', 'exponent', 'none').
  """
  import pandas as pd
  import numpy as np
  from scipy.stats import yeojohnson

  out = df.copy()
  numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
  # Exclude boolean 0/1 columns from transformation
  cols_eligible = []
  for c in numeric_cols:
    uniq = set(out[c].dropna().unique())
    if uniq.issubset({0, 1}):
      continue
    cols_eligible.append(c)

  if columns is not None and len(columns) > 0:
    to_process = [c for c in columns if c in cols_eligible]
  else:
    to_process = list(cols_eligible)

  report = []
  transformations = {}  # output_column_name -> lambda (float) or transform name (str)

  for col in to_process:
    x = out[col].astype(float)
    x_valid = x.dropna()
    if len(x_valid) < 2:
      continue
    skew_orig = x.skew()
    candidates = {}  # name -> series (aligned to out.index)
    yj_lambda = None  # set when Yeo-Johnson is computed

    # Yeo-Johnson works for any real values
    try:
      yj_vals, yj_lambda = yeojohnson(x_valid)
      yj_series = x.copy()
      yj_series.loc[x_valid.index] = yj_vals
      candidates["yeojohnson"] = yj_series
    except Exception:
      pass

    if skew_orig > 0:
      # Positive skew: sqrt, cbrt, ln (where applicable)
      if (x_valid >= 0).all():
        sqrt_vals = np.sqrt(x_valid)
        if np.isfinite(sqrt_vals).all():
          s = x.copy()
          s.loc[x_valid.index] = sqrt_vals
          candidates["sqrt"] = s
      cbrt_vals = np.cbrt(x_valid)
      if np.isfinite(cbrt_vals).all():
        s = x.copy()
        s.loc[x_valid.index] = cbrt_vals
        candidates["cbrt"] = s
      if (x_valid > 0).all():
        ln_vals = np.log(x_valid)
        if np.isfinite(ln_vals).all():
          s = x.copy()
          s.loc[x_valid.index] = ln_vals
          candidates["ln"] = s
    else:
      # Negative skew: square, cube, exponent (x**4)
      sq_vals = np.square(x_valid)
      if np.isfinite(sq_vals).all():
        s = x.copy()
        s.loc[x_valid.index] = sq_vals
        candidates["square"] = s
      cube_vals = np.power(x_valid, 3)
      if np.isfinite(cube_vals).all():
        s = x.copy()
        s.loc[x_valid.index] = cube_vals
        candidates["cube"] = s
      exp_vals = np.power(x_valid, 4)
      if np.isfinite(exp_vals).all():
        s = x.copy()
        s.loc[x_valid.index] = exp_vals
        candidates["exponent"] = s

    if not candidates:
      continue

    best_name = None
    best_skew = abs(skew_orig)
    best_series = None
    for name, ser in candidates.items():
      sk = ser.skew()
      if pd.isna(sk):
        continue
      if abs(sk) < best_skew:
        best_skew = abs(sk)
        best_name = name
        best_series = ser

    if best_series is None:
      best_name = "none"
      best_skew = abs(skew_orig)
      best_series = x

    new_col = f"{col}_normalized"
    out[new_col] = best_series
    skew_new = best_series.skew()
    report.append((col, skew_orig, best_name, skew_new))

    # Record exact transformation for this column: lambda if yeojohnson, else transform name
    if keep_originals:
      target_col = new_col
    else:
      target_col = col
    if best_name == "yeojohnson" and yj_lambda is not None:
      transformations[target_col] = float(yj_lambda)
    else:
      transformations[target_col] = best_name

    if not keep_originals:
      out = out.drop(columns=[col])
      out = out.rename(columns={new_col: col})

  if verbose and report:
    print("=== Normalize (skew-reduction) report ===")
    for col, sk_orig, transform, sk_new in report:
      if keep_originals:
        target_col = f"{col}_normalized"
      else:
        target_col = col
      exact = transformations.get(target_col, transform)
      exact_str = f"lambda={float(exact):.4f}" if not isinstance(exact, str) else exact
      print(f"  {col}: skew {sk_orig:.4f} -> {exact_str} -> skew {sk_new:.4f}  (saved as '{target_col}')")
    print()

  return pd.DataFrame(out), transformations


def manage_outliers(df, epsilon=0.5, min_samples=5, columns=None, drop_outliers=False, verbose=True):
  """
  Identify outlier rows using DBSCAN (noise points are treated as outliers).
  Optionally drop outlier rows or only report counts and which features drive outlier status.

  Parameters
  ----------
  df : pandas.DataFrame
      Input dataframe.
  epsilon : float, default 0.5
      DBSCAN maximum distance between two samples for one to be in the neighborhood of the other.
      Numeric features are standardized before clustering, so epsilon is in scaled units.
  min_samples : int, default 5
      DBSCAN minimum number of samples in a neighborhood for a core point.
  columns : list of str or None, optional
      Numeric columns to use for clustering. If None, all numeric columns are used.
  drop_outliers : bool, default False
      If True, remove rows that DBSCAN labels as noise (outliers). If False, return the
      full dataframe and only report outlier information when verbose=True.
  verbose : bool, default True
      If True, print the number of outliers and, for each outlier row, which features
      contribute most to its distance from the nearest cluster (helps explain why it was flagged).

  Returns
  -------
  pandas.DataFrame
      If drop_outliers=True, dataframe with outlier rows removed. Otherwise, the original
      dataframe unchanged.
  """
  import pandas as pd
  import numpy as np
  from sklearn.cluster import DBSCAN
  from sklearn.preprocessing import StandardScaler

  out = df.copy()
  numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
  if columns is not None and len(columns) > 0:
    numeric_cols = [c for c in columns if c in numeric_cols]
  if not numeric_cols:
    if verbose:
      print("manage_outliers: no numeric columns to cluster.")
    return pd.DataFrame(out)

  # Use only rows with no missing values in the clustering columns
  X = out[numeric_cols].dropna(how="any")
  if len(X) < 2:
    if verbose:
      print("manage_outliers: not enough rows with complete numeric data.")
    return pd.DataFrame(out)

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  clusterer = DBSCAN(eps=epsilon, min_samples=min_samples)
  labels = clusterer.fit_predict(X_scaled)

  outlier_mask = labels == -1
  n_outliers = int(outlier_mask.sum())
  outlier_indices = X.index[outlier_mask].tolist()

  if verbose:
    print("=== Outlier report (DBSCAN) ===")
    print(f"  Epsilon: {epsilon}, min_samples: {min_samples}")
    print(f"  Rows used for clustering: {len(X)} (rows with missing values in clustering columns excluded)")
    print(f"  Number of outlier rows (noise): {n_outliers}")
    if n_outliers > 0:
      # Feature contribution: for each outlier, distance to nearest cluster centroid by feature
      core_mask = labels >= 0
      if core_mask.any():
        unique_labels = np.unique(labels[core_mask])
        centroids = np.array([X_scaled[labels == k].mean(axis=0) for k in unique_labels])
        for idx in outlier_indices[:20]:  # cap at 20 to avoid huge output
          row = X_scaled[X.index == idx][0]
          dists = np.linalg.norm(centroids - row, axis=1)
          nearest_idx = np.argmin(dists)
          diff = np.abs(row - centroids[nearest_idx])
          order = np.argsort(diff)[::-1]
          top = [f"{numeric_cols[j]} ({diff[j]:.2f})" for j in order[:5]]
          print(f"  Outlier row {idx}: top contributing features (scaled) — {', '.join(top)}")
        if n_outliers > 20:
          print(f"  ... and {n_outliers - 20} more outlier rows.")
      else:
        print("  (No core points; cannot compute feature contributions.)")
    print()

  if drop_outliers and n_outliers > 0:
    out = out.drop(index=outlier_indices)

  return pd.DataFrame(out)