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