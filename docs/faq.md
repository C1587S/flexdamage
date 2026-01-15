# FlexDamage Notes [NEed to add this somewhere else]

## Configuration Questions

### difference between `regional.aggregation_level` and `data.aggregation`?

*   **`data.aggregation`**: Controls how the **input data** is processed before it enters the estimation pipeline.
    *   Example: You have pixel-level data but want to run analysis at the country level.
    *   Settings: `dims: ["iso3", "year"]`, `method: "mean"`, `weights: "pop"`.
    *   Result: The pipeline will group your raw data by `iso3` and `year` and compute population-weighted means.

*   **`regional.aggregation_level`**: Controls the **unit of analysis** for the Regional Estimation step.
    *   Example: `aggregation_level: "iso3"`.
    *   Meaning: The `RegionalEstimator` looks for a column named `"iso3"` to identify distinct regions. It will produce one set of polynomial coefficients (alphas, betas) for each unique value in this column.

**Consistency Check**:

Check this in config file

*   If your `data.aggregation` collapses the data to `["iso3", "year"]`, then your dataframe will contain an `"iso3"` column.
*   Therefore, your `regional.aggregation_level` **must** be `"iso3"` (or whatever name you gave that dimension) so the estimator knows which column to use as the region identifier.

If you set `regional.aggregation_level: "pixel_id"` but your data was already aggregated to `"iso3"`, the estimator will fail because it cannot find the `"pixel_id"` column.

## Constraints

### enforce convexity (e.g., beta >= 0)?

You can define constraints in the `regional` section of your config.

```yaml
regional:
  constraints:
    - param: "beta"
      type: "convexity"  # Enforces beta >= 0
```

There is an option to use explicit formulas:

```yaml
regional:
  constraints:
    - param: "beta"
      formula: "beta >= 0"
```

If the unconstrained estimate violates this (e.g., `beta = -0.05`), the model will re-run with `beta` fixed to `0` (linear fit).
