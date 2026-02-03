# type: ignore

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def univariate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate univariate statistical analysis and visualizations for a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze
    
    Returns
    -------
    pd.DataFrame
        Summary statistics for each column
    """

    df_results = pd.DataFrame(columns=["Data Type", "Count", "Missing", "Unique", "Mode", "Min", "Q1", "Median",
                                        "Q3", "Max", "Mean", "Std", "Skew", "Kurt"])

    for col in df.columns:
        df_results.loc[col, "Data Type"] = df[col].dtype
        df_results.loc[col, "Count"] = df[col].count()
        df_results.loc[col, "Missing"] = df[col].isna().sum()
        df_results.loc[col, "Unique"] = df[col].nunique()
        
        # Handle mode safely
        mode_values = df[col].mode()
        df_results.loc[col, "Mode"] = mode_values[0] if len(mode_values) > 0 else None

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
                # Create stacked plot: box plot on top, histogram with KDE underneath
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                                gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.3})
                
                # Box plot on top
                sns.boxplot(data=df, y=col, ax=ax1)
                ax1.set_title(f'Box Plot and Distribution for {col}')
                ax1.set_xlabel('')
                ax1.set_ylabel(col)
                
                # Histogram with KDE overlay underneath
                sns.histplot(data=df, x=col, kde=True, ax=ax2)
                ax2.set_xlabel(col)
                ax2.set_ylabel('Frequency')
                
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
            if total > 0:
                for p in ax.patches:
                    height = p.get_height()
                    percentage = (height / total) * 100
                    ax.text(p.get_x() + p.get_width() / 2., height,
                            f'{percentage:.1f}%',
                            ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()

    return df_results

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with no predictive power.
    
    Removes columns where:
    - All values are identical (0 or 1 unique value)
    - All values are unique and the column is non-numeric (e.g., ID columns)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    pd.DataFrame
        DataFrame with low-predictive-power columns removed
    """
    cols_to_keep = []
    for col in df.columns:
        n_unique = df[col].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        # Keep column if it has multiple unique values
        if n_unique > 1:
            # Drop only if all values are unique AND non-numeric
            if not (n_unique == len(df) and not is_numeric):
                cols_to_keep.append(col)
    
    return df[cols_to_keep]

def bin_categories(df: pd.DataFrame, min_count:int=15, threshold:float=None) -> pd.DataFrame:
    """
    Bin rare categorical values into an 'Other' category.
    
    For each categorical column, identifies values that appear infrequently
    and groups them into an 'Other' category to reduce dimensionality.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_count : int, optional
        Minimum number of occurrences for a category to remain separate.
        Categories appearing fewer times will be binned as 'Other'. Default is 10.
    threshold : float, optional
        Alternative to min_count: minimum percentage (0-1) of total rows
        for a category to remain separate. If provided, overrides min_count.
        For larger datasets, consider using lower thresholds (e.g., 0.01 for 1%).
        Default is None.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rare categories binned into 'Other'
    """
    import pandas as pd

    for col in df.columns:
        if df[col].dtype == 'object':
            if threshold is not None:
                # Use percentage threshold
                value_counts = df[col].value_counts(normalize=True)
                to_bin = value_counts[value_counts < threshold].index
            else:
                # Use absolute count threshold
                value_counts = df[col].value_counts()
                to_bin = value_counts[value_counts < min_count].index
            
            # Only replace if there are categories to bin
            if len(to_bin) > 0:
                df[col] = df[col].replace(to_bin, 'Other')
    
    return df

def missing_data_diagnostics(df: pd.DataFrame, show_plots: bool = True) -> pd.DataFrame:
    """
    Analyze missing data patterns and test for missing data mechanisms.
    
    Generates comprehensive diagnostics including:
    - Missing value counts and percentages per column
    - Rows with missing data count
    - Visual heatmap of missing data patterns
    - Correlation analysis to assess MCAR (Missing Completely At Random)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze
    show_plots : bool, optional
        Whether to display visualization plots. Default is True.
    
    Returns
    -------
    pd.DataFrame
        Diagnostic report with columns:
        - Column: Column name
        - Missing_Count: Number of missing values
        - Missing_Percent: Percentage of missing values
        - Data_Type: Column data type
    
    Notes
    -----
    Missing data mechanisms:
    - MCAR (Missing Completely At Random): Missingness independent of any data
    - MAR (Missing At Random): Missingness related to observed data
    - MNAR (Missing Not At Random): Missingness related to unobserved data
    
    This function helps identify patterns suggesting MAR or MNAR through
    correlation analysis. High correlations between missingness indicators
    suggest data is not MCAR.
    """
    # Calculate missing data statistics
    missing_counts = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    # Create diagnostic DataFrame
    diagnostics = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_counts.values,
        'Missing_Percent': missing_percent.values,
        'Data_Type': df.dtypes.values
    })
    
    # Filter to only columns with missing data
    diagnostics = diagnostics[diagnostics['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)
    
    if len(diagnostics) == 0:
        print("✓ No missing data detected in the dataset.")
        return diagnostics
    
    # Print summary
    print("=" * 70)
    print("MISSING DATA DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns with missing data: {len(diagnostics)}")
    print(f"Rows with any missing data: {df.isnull().any(axis=1).sum()}")
    print(f"Total missing values: {df.isnull().sum().sum()}")
    print(f"Overall missing percentage: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%")
    
    print("\n" + "=" * 70)
    print("MISSING DATA BY COLUMN")
    print("=" * 70)
    print(diagnostics.to_string(index=False))
    
    if show_plots and len(diagnostics) > 0:
        # Create missing data heatmap
        plt.figure(figsize=(12, 8))
        
        # Only plot columns with missing data
        cols_with_missing = diagnostics['Column'].tolist()
        missing_data = df[cols_with_missing].isnull()
        
        # If too many rows, sample for visualization
        if len(df) > 500:
            missing_data_sample = missing_data.sample(n=500, random_state=42)
            title_suffix = " (500 row sample)"
        else:
            missing_data_sample = missing_data
            title_suffix = ""
        
        sns.heatmap(missing_data_sample.T, cbar=True, cmap='viridis', 
                    yticklabels=True, xticklabels=False)
        plt.title(f'Missing Data Heatmap{title_suffix}')
        plt.xlabel('Rows')
        plt.ylabel('Columns')
        plt.tight_layout()
        plt.show()
        
        # Correlation analysis for MCAR test
        print("\n" + "=" * 70)
        print("MISSING DATA MECHANISM ANALYSIS")
        print("=" * 70)
        
        # Create missingness indicator matrix
        missing_indicators = df[cols_with_missing].isnull().astype(int)
        
        if len(cols_with_missing) > 1:
            # Calculate correlation between missingness indicators
            missing_corr = missing_indicators.corr()
            
            # Check for high correlations (excluding diagonal)
            high_corr_pairs = []
            for i in range(len(missing_corr.columns)):
                for j in range(i+1, len(missing_corr.columns)):
                    corr_val = missing_corr.iloc[i, j]
                    if abs(corr_val) > 0.3:  # Threshold for noteworthy correlation
                        high_corr_pairs.append((missing_corr.columns[i], 
                                               missing_corr.columns[j], 
                                               corr_val))
            
            if len(high_corr_pairs) > 0:
                print("\n⚠ HIGH CORRELATIONS between missing data patterns detected:")
                print("   This suggests data may be MAR or MNAR (not MCAR)\n")
                for col1, col2, corr in high_corr_pairs:
                    print(f"   {col1} ↔ {col2}: {corr:.3f}")
            else:
                print("\n✓ LOW CORRELATIONS between missing data patterns.")
                print("   Data appears consistent with MCAR (Missing Completely At Random).")
            
            # Visualize correlation matrix
            if len(cols_with_missing) >= 2:
                plt.figure(figsize=(10, 8))
                sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0,
                           vmin=-1, vmax=1, square=True, linewidths=1)
                plt.title('Correlation Between Missingness Patterns')
                plt.tight_layout()
                plt.show()
        else:
            print("\nOnly one column with missing data - cannot assess correlations.")
            print("Consider examining relationship with other variables manually.")
    
    print("\n" + "=" * 70)
    
    return diagnostics

def manage_dates(df: pd.DataFrame, startdate=None, enddate=None) -> pd.DataFrame:
    """
    Converts date columns to datetime format and extracts date features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    startdate : str or datetime, optional
        Start date to calculate days difference from
    enddate : str or datetime, optional
        End date to calculate days difference to
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with extracted date features
    """
    df = df.copy()
    
    # Identify and convert date columns
    date_columns = []
    for col in df.columns:
        # Skip if column is already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue
            
        # Try to convert to datetime
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            try:
                # Attempt conversion with error handling
                converted = pd.to_datetime(df[col], errors='coerce')
                # Check if at least 50% of non-null values were successfully converted
                if converted.notna().sum() / df[col].notna().sum() > 0.5:
                    df[col] = converted
                    date_columns.append(col)
            except:
                pass
    
    # Extract features from identified date columns
    for col in date_columns:
        # Extract date components
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_weekday'] = df[col].dt.weekday  # Monday=0, Sunday=6
        df[f'{col}_day_of_week'] = df[col].dt.day_name()
        
        # Extract hour if time component exists
        if df[col].dt.hour.notna().any() and (df[col].dt.hour != 0).any():
            df[f'{col}_hour'] = df[col].dt.hour
        
        # Calculate days difference from startdate
        if startdate is not None:
            start = pd.to_datetime(startdate)
            df[f'{col}_days_from_start'] = (df[col] - start).dt.days
        
        # Calculate days difference to enddate
        if enddate is not None:
            end = pd.to_datetime(enddate)
            df[f'{col}_days_to_end'] = (end - df[col]).dt.days
    
    return df

def handle_missing_data(df: pd.DataFrame, 
                        drop_threshold: float = 0.9,
                        impute_strategy: str = 'mean',
                        verbose: bool = False) -> pd.DataFrame:
    """
    Handle missing data through threshold-based dropping and imputation.
    
    First drops columns and rows exceeding the missing data threshold,
    then imputes remaining missing values using the specified strategy.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potential missing values
    drop_threshold : float, optional
        Proportion threshold (0-1) for dropping columns/rows.
        Columns with missing proportion > threshold are dropped.
        After column dropping, rows with missing proportion > threshold are dropped.
        Default is 0.9 (90%).
    impute_strategy : str, optional
        Strategy for imputing remaining missing values:
        - 'mean': Replace with column mean (numeric only)
        - 'median': Replace with column median (numeric only)
        - 'mode': Replace with most frequent value (all types)
        - 'drop': Drop remaining rows with any missing values
        - 'knn': Use KNN imputation (requires scikit-learn)
        - 'iterative': Use iterative imputation (requires scikit-learn)
        Default is 'mean'.
    verbose : bool, optional
        Whether to print detailed diagnostic information during processing.
        If True, shows which columns/rows are dropped and imputation details.
        Default is False.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing data handled according to specifications
    
    Raises
    ------
    ValueError
        If impute_strategy is not one of the valid options
    ImportError
        If 'knn' or 'iterative' strategy is used but scikit-learn is not installed
    
    Examples
    --------
    >>> # Basic usage with mean imputation
    >>> df_clean = handle_missing_data(df)
    
    >>> # Stricter dropping threshold with median imputation
    >>> df_clean = handle_missing_data(df, drop_threshold=0.5, impute_strategy='median')
    
    >>> # Verbose mode to see what's happening
    >>> df_clean = handle_missing_data(df, verbose=True)
    """
    valid_strategies = ['mean', 'median', 'mode', 'drop', 'knn', 'iterative']
    if impute_strategy not in valid_strategies:
        raise ValueError(f"impute_strategy must be one of {valid_strategies}")
    
    if verbose:
        print("=" * 70)
        print("MISSING DATA HANDLING")
        print("=" * 70)
        print(f"\nInitial shape: {df.shape}")
        print(f"Initial missing values: {df.isnull().sum().sum()}")
    
    # Work on a copy to avoid modifying original
    df_clean = df.copy()
    
    # Step 1: Drop columns with high missing proportion
    missing_by_col = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = missing_by_col[missing_by_col > drop_threshold].index.tolist()
    
    if len(cols_to_drop) > 0:
        df_clean = df_clean.drop(columns=cols_to_drop)
        if verbose:
            print(f"\n✗ Dropped {len(cols_to_drop)} columns with >{drop_threshold*100:.0f}% missing:")
            for col in cols_to_drop:
                print(f"   - {col} ({missing_by_col[col]*100:.1f}% missing)")
    elif verbose:
        print(f"\n✓ No columns exceed {drop_threshold*100:.0f}% missing threshold")
    
    # Step 2: Drop rows with high missing proportion
    missing_by_row = df_clean.isnull().sum(axis=1) / len(df_clean.columns)
    rows_to_drop = missing_by_row[missing_by_row > drop_threshold].index
    
    if len(rows_to_drop) > 0:
        df_clean = df_clean.drop(index=rows_to_drop)
        if verbose:
            print(f"\n✗ Dropped {len(rows_to_drop)} rows with >{drop_threshold*100:.0f}% missing")
    elif verbose:
        print(f"✓ No rows exceed {drop_threshold*100:.0f}% missing threshold")
    
    # Step 3: Impute remaining missing values
    remaining_missing = df_clean.isnull().sum().sum()
    
    if remaining_missing == 0:
        if verbose:
            print(f"\n✓ No missing values remaining after threshold-based dropping")
            print(f"\nFinal shape: {df_clean.shape}")
        return df_clean
    
    if verbose:
        print(f"\n→ {remaining_missing} missing values remaining across {df_clean.isnull().any().sum()} columns")
        print(f"→ Applying '{impute_strategy}' imputation strategy...")
    
    if impute_strategy == 'drop':
        df_clean = df_clean.dropna()
        if verbose:
            print(f"   Dropped {len(df) - len(df_clean)} rows with any missing values")
    
    elif impute_strategy in ['mean', 'median', 'mode']:
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if impute_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_value = df_clean[col].mean()
                    df_clean[col].fillna(fill_value, inplace=True)
                    if verbose:
                        print(f"   {col}: filled with mean = {fill_value:.2f}")
                
                elif impute_strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_value = df_clean[col].median()
                    df_clean[col].fillna(fill_value, inplace=True)
                    if verbose:
                        print(f"   {col}: filled with median = {fill_value:.2f}")
                
                elif impute_strategy == 'mode' or not pd.api.types.is_numeric_dtype(df_clean[col]):
                    # Use mode for categorical or if mode strategy specified
                    mode_values = df_clean[col].mode()
                    if len(mode_values) > 0:
                        fill_value = mode_values[0]
                        df_clean[col].fillna(fill_value, inplace=True)
                        if verbose:
                            print(f"   {col}: filled with mode = {fill_value}")
    
    elif impute_strategy in ['knn', 'iterative']:
        try:
            from sklearn.impute import KNNImputer, IterativeImputer
        except ImportError:
            raise ImportError(f"'{impute_strategy}' strategy requires scikit-learn. "
                            "Install it with: pip install scikit-learn")
        
        # Separate numeric and categorical columns
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with advanced method
        if len(numeric_cols) > 0 and df_clean[numeric_cols].isnull().any().any():
            if impute_strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:  # iterative
                imputer = IterativeImputer(random_state=42, max_iter=10)
            
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            if verbose:
                print(f"   Applied {impute_strategy} imputation to {len(numeric_cols)} numeric columns")
        
        # Impute categorical columns with mode
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_values = df_clean[col].mode()
                if len(mode_values) > 0:
                    df_clean[col].fillna(mode_values[0], inplace=True)
                    if verbose:
                        print(f"   {col}: filled with mode (categorical)")
    
    if verbose:
        final_missing = df_clean.isnull().sum().sum()
        print(f"\n✓ Imputation complete")
        print(f"   Missing values after imputation: {final_missing}")
        print(f"\nFinal shape: {df_clean.shape}")
        print("=" * 70)
    
    return df_clean