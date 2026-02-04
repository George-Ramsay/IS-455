# Data Wrangling & Feature Engineering Assignment

## Section A — Concept Questions (Multiple Choice)

### 1) Which statement best describes the goal of data wrangling in a machine learning pipeline?
- A. To standardize all datasets into a single universal format  
- B. To prepare raw data so it is accurate, consistent, and usable for modeling  
- C. To remove as many rows as possible to simplify modeling  
- D. To engineer only new predictive features  

**Answer:** B
---

### 2) Why is categorical binning (grouping rare categories into “Other”) often necessary?
- A. To eliminate all categorical variables  
- B. To satisfy minimum frequency assumptions and improve model stability  
- C. To make categories easier for humans to read  
- D. To increase the number of unique categories  

**Answer:** B
---

### 3) Why are raw date values (e.g., "2024-03-15") rarely useful directly in predictive models?
- A. Dates contain multiple latent components (year, month, weekday, etc.)  
- B. Dates violate most machine learning assumptions  
- C. Dates are categorical variables by default  
- D. Dates cannot be converted into numeric values  

**Answer:** A
---

### 4) Which situation most strongly motivates applying a logarithmic transformation?
- A. A feature has values evenly distributed around zero  
- B. A feature is heavily right-skewed with extreme large values  
- C. A feature is already normally distributed  
- D. A feature contains many zeros  

**Answer:** B
---

### 5) Which type of missing data often requires domain knowledge to address?
- A. MCAR (Missing Completely at Random)  
- B. MAR (Missing at Random)  
- C. Structurally missing data  
- D. MNAR (Missing Not at Random)  

**Answer:** D
---

### 6) Why might advanced imputation techniques (e.g., KNN or Iterative Imputation) not always be appropriate?
- A. They can introduce bias into the dataset  
- B. They cannot handle categorical variables  
- C. They are computationally expensive and increase pipeline complexity  
- D. They are less accurate than simple methods  

**Answer:** C
---

### 7) Why does the Empirical Rule (Z-Score method) perform poorly on skewed data?
- A. It assumes a normal distribution  
- B. It cannot compute standard deviation  
- C. It ignores extreme values  
- D. It identifies too few outliers  

**Answer:** A
---

### 8) What is a key advantage of the IQR (Tukey) method over Z-scores?
- A. It works only for normally distributed data  
- B. It is unaffected by extreme outliers  
- C. It requires fewer computations  
- D. It identifies multivariate outliers  

**Answer:** B
---

### 9) Why are clustering-based methods like DBSCAN useful for outlier detection?
- A. They assume all features are normally distributed  
- B. They require labeled training data  
- C. They only work on small datasets  
- D. They identify outliers across multiple features simultaneously  

**Answer:** D
---

## Section B — Coding Tasks

### 10) Function: `wrangle_basic(df)`
**Goal:** Create a function named `wrangle_basic(df)` that cleans categorical text fields to eliminate data quality issues caused by inconsistent data entry.  
Example: `"UT"`, `"ut"`, `"Utah"`, and `"utah"` should all be relabeled to the same exact string.

**Function requirements:**
- Create cleaned versions of problematic columns (append `"_clean"` to column names; do not overwrite originals)
- Ensure that semantically identical values are represented consistently
- Preserve the original number of rows
- **Hint:** Start by exploring the unique values in categorical columns to identify patterns of inconsistency.

**Check question (do not answer here):**
- After running your function, how many rows are labeled as `"failed"` in `delivery_status_clean`?

---

### 11) Function: `add_datetime_features(df)`
**Goal:** Create a function named `add_datetime_features(df)` that converts messy datetime strings into usable datetime objects and creates time-based analytical features.

**Function requirements:**
- Parse datetime information from text columns that contain date/time data
- Handle inconsistent formatting (different date orders, 12/24-hour time, timezone tokens)
- Successfully parse at least these columns:
  - `stop_datetime_raw`
  - `scheduled_window_start_raw`
- Create parsed versions with appropriate naming to distinguish from originals
- Engineer time-based features for analysis:
  - Day of week indicator (numeric, starting with Monday)
  - Weekend indicator (binary: weekend vs weekday)
  - Delivery lateness metric that compares actual arrival time to the scheduled window
- **Hint:** The `actual_arrival_min` column contains minutes from midnight. Your lateness metric should be comparable to this scale and never negative.

**Check question (do not answer here):**
- After running your function (on the output of Question 10), what is the mean of your lateness metric, rounded to 2 decimals?

---

### 12) Function: `bin_rare_categories(df, cols=None, min_prop=0.05, suffix='_binned')`
**Goal:** Create a function named `bin_rare_categories(df, cols=None, min_prop=0.05, suffix='_binned')` that consolidates infrequent categories to reduce cardinality.

**Function requirements:**
- Accept flexible column specification:
  - If `cols=None` (default), process all categorical columns in the DataFrame
  - If `cols` is a string, process that single column
  - If `cols` is a list, process those specific columns
- Reduce category cardinality by grouping rare values:
  - Calculate the frequency of each category as a proportion of total rows
  - Categories that appear too infrequently should be consolidated into a single group
  - The threshold for `"infrequent"` is controlled by the `min_prop` parameter (default: `0.05`)
- Create new binned columns:
  - Add a suffix to distinguish binned columns from originals (controlled by `suffix`, default: `'_binned'`)
- Preserve all rows (no filtering)
- **Hint:** Common practice is to label consolidated rare categories as `"Other"`.

**Check question (do not answer here):**
- After running your function on `delivery_zone_clean` (from Question 11 output) with default parameters, how many unique categories exist in the resulting binned column?

---

### 13) Function: `transform_skew(df, features=None, suffix='_skewfix')`
**Goal:** Create a function named `transform_skew(df, features=None, suffix='_skewfix')` that reduces skew in numeric columns by automatically selecting the best transformation.

**Function requirements:**
- Accept flexible feature specification:
  - If `features=None` (default), process all numeric non-boolean columns
  - If `features` is a string, process that single column
  - If `features` is a list, process those specific columns
- Handle data with varied characteristics:
  - Some columns may contain negative values, zeros, or missing values
  - Your transformations must handle these gracefully without errors
  - Consider using a shift strategy for transforms that require non-negative inputs
- Evaluate multiple transformation strategies:
  - Test several monotonic transformations including no transformation as a baseline
  - Include both traditional power transformations AND advanced statistical transformations
  - Research the **Yeo-Johnson transformation** (handles negative values natively)
  - Select the transformation that minimizes the absolute value of skewness
  - Implement a consistent tie-breaking strategy if multiple transformations produce similar results
- Create new columns with appropriate naming:
  - Use the `suffix` parameter to distinguish transformed columns from originals (default: `'_skewfix'`)
- Preserve all rows and original columns

**Hints:**
- Skewness can be calculated with `.skew(skipna=True)`
- For transforms requiring non-negative inputs: consider shifting data so minimum becomes zero
- The Yeo-Johnson transformation is available in `scipy.stats` and works with negative values
- Common power transformations: logarithmic (`log1p`), square root, cube root
- When comparing transformations, focus on which produces skewness closest to zero

**Check question (do not answer here):**
- After running your function on `distance_from_prev_mi` (from Question 12 output) with default parameters, what is the skewness of the transformed column, rounded to 3 decimals?

---

### 14) Function: `impute_missing(df, features=None, group_cols=None)`
**Goal:** Create a function named `impute_missing(df, features=None, group_cols=None)` that fills in missing values without losing any data rows.

**Function requirements:**
- Accept flexible feature specification:
  - If `features=None` (default), impute all columns that contain missing values
  - If `features` is a string, impute that single column
  - If `features` is a list, impute only those specific columns
- Accept flexible grouping specification:
  - If `group_cols=None` (default), automatically select logical grouping columns from the data
    - Look for cleaned categorical columns that represent natural groups (e.g., location, category indicators)
  - If `group_cols` is a list, use those specific columns for grouping
- Implement intelligent imputation strategies:
  - Use different strategies for numeric vs categorical data
  - Apply group-based imputation first (calculate statistics within groups)
  - Fall back to global imputation when group-based values are unavailable
- Ensure deterministic results (same input always produces same output)
- Preserve all rows:
  - Never drop rows or columns
  - All missing values must be filled

**Hints:**
- For numeric data: central tendency measures like median work well
- For categorical data: most frequent values (mode) are appropriate
- Group-based imputation: calculate statistics separately for each group, then use those to fill within-group missing values
- Fallback strategy: if a group has all missing values, use the overall (global) statistic

**Check question (do not answer here):**
- After running your function on the output of Question 13, what is the mean of `cargo_temp_f` for records where `hub_clean == "denver-east"`, rounded to 2 decimal places?

---

### 15) Function: `cap_outliers_iqr(df, cols=None)`
**Goal:** Create a function named `cap_outliers_iqr(df, cols=None)` that handles extreme values using a statistical outlier detection method.

**Function requirements:**
- Accept flexible column specification:
  - If `cols=None` (default), process all numeric non-boolean columns
  - If `cols` is a string, process that single column
  - If `cols` is a list, process those specific columns
- Implement outlier detection and capping:
  - Use a robust statistical method based on the interquartile range (IQR)
  - Identify values that fall outside a reasonable range
  - Instead of removing outliers, adjust them to boundary values (winsorization)
  - Preserve all data rows while limiting extreme influence
- Apply Tukey's fence method:
  - Calculate quartiles and the interquartile range
  - Define bounds using a standard multiplier of the IQR
  - Cap values that exceed these bounds
- Preserve data integrity:
  - Never drop rows or columns
  - Ensure deterministic results

**Hints:**
- IQR (Interquartile Range) = Q3 - Q1
- Tukey's fences use the multiplier 1.5 for outlier detection
- Quartiles can be calculated with `.quantile()`
- "Capping" or "winsorization" means setting values to the boundary rather than removing them
- The `.clip()` method can limit values to a range

**Check question (do not answer here):**
- After running your function on the output of Question 14 with default parameters, what is the maximum value of `service_time_min` rounded to 4 decimal places?

---

## Section C — Business Analysis Report

### 16) Comprehensive Delivery Performance Report (using the fully cleaned dataset)
**Goal:** Create a comprehensive business analysis report that demonstrates the value of data cleaning by revealing actionable insights about delivery performance.

**Task:** Using the fully cleaned dataset (from the prior question assuming it has undergone each of the prior steps), generate an analysis report with the following components:

#### A) Hub Performance Analysis
- Calculate the delivery success rate (percentage of deliveries with status `"delivered"`) for each hub
- Sort hubs from highest to lowest success rate
- Display results in a formatted table showing `success_rate_pct` for each `hub_clean` value

#### B) Priority Level Risk Analysis
- Calculate the delivery failure rate (percentage of deliveries with status `"failed"`) for each priority level
- Sort priority levels from highest to lowest failure rate
- Display results in a formatted table showing `failure_rate_pct` for each `priority_level_clean` value

#### C) Visualization
- Create a 2-panel figure (1 row, 2 columns, size 14x5 inches) using matplotlib subplots
- **Left panel:** Horizontal bar chart showing success rate by hub (**steelblue** color)
  - X-axis: `"Success Rate (%)"`
  - Y-axis: `"Hub"`
  - Title: `"Delivery Success Rate by Hub"` (bold, size 13)
  - Add percentage labels on each bar
- **Right panel:** Horizontal bar chart showing failure rate by priority (**coral** color)
  - X-axis: `"Failure Rate (%)"`
  - Y-axis: `"Priority Level"`
  - Title: `"Delivery Failure Rate by Priority"` (bold, size 13)
  - Add percentage labels on each bar

#### D) Business Insights Summary
- Print a formatted section titled: `ACTIONABLE BUSINESS INSIGHTS`
- Include:
  - **Hub Performance Analysis:** Identify best and worst performing hubs, calculate the performance gap, and recommend investigating underperforming locations
  - **Priority Level Risk Analysis:** Identify highest and lowest risk priority levels, calculate the risk differential, and recommend resource allocation strategies
  - **Data Cleaning Impact:** Explain how cleaning enabled this analysis  
    - Example format:  
      - Before cleaning: 14+ inconsistent hub names prevented reliable analysis.  
      - After cleaning: 3 standardized hubs enable actionable insights.

#### Output Format Requirements
- Code should print clearly formatted tables and insights
- Use equal sign separators (`=`) **60 characters** wide
- Use emoji icons (📊, 📦, ✅) to organize the report sections

**Hint:** You should be able to plug these requirements directly into an AI agent and it will write the code for you.

**Check question (do not answer here):**
- What is the delivery success rate (percentage of `"delivered"` status) for the hub with the highest success rate, rounded to 1 decimal place?
