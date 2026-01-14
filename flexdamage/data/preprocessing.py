import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from ..config.schema import DataConfig, DataTransformation, DataAggregation

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles data transformations and aggregation before estimation.
    """
    def __init__(self, config: DataConfig):
        self.config = config
        
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations and aggregation to the dataframe.
        """
        df = df.copy()
        
        # 1. Transformations
        if self.config.transformations:
            for trans in self.config.transformations:
                self._apply_transformation(df, trans)
                
        # 2. Aggregation
        if self.config.aggregation:
            df = self._apply_aggregation(df, self.config.aggregation)
            
        return df
        
    def _apply_transformation(self, df: pd.DataFrame, trans: DataTransformation):
        col = self.config.columns.get(trans.variable, trans.variable)
        if col not in df.columns:
            logger.warning(f"Transformation target column {col} not found. Skipping.")
            return

        logger.info(f"Applying transformation {trans.method} to {col}")
        
        if trans.method == "scale":
            if trans.value is None:
                raise ValueError("Value required for scale transformation")
            df[col] = df[col] * trans.value
        elif trans.method == "offset":
            if trans.value is None:
                raise ValueError("Value required for offset transformation")
            df[col] = df[col] + trans.value
        elif trans.method == "log":
            # Avoid log(0) or negative
            if (df[col] <= 0).any():
                logger.warning(f"Column {col} has non-positive values. Log transform might fail or produce NaNs.")
            df[col] = np.log(df[col])
            
    def _apply_aggregation(self, df: pd.DataFrame, agg: DataAggregation):
        logger.info(f"Aggregating data by {agg.dims} using {agg.method}")
        
        # Map dim names if they are keys in columns map
        group_cols = [self.config.columns.get(d, d) for d in agg.dims]
        
        # Check if all group columns exist
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Aggregation group columns missing: {missing}")
            
        # Identify numeric columns to aggregate
        # Exclude group cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        agg_cols = [c for c in numeric_cols if c not in group_cols]
        
        if agg.method == "mean":
            if agg.weights:
                w_col = self.config.columns.get(agg.weights, agg.weights)
                if w_col not in df.columns:
                     raise ValueError(f"Weight column {w_col} missing for aggregation.")
                
                # Weighted Mean
                def weighted_mean(x):
                    w = df.loc[x.index, w_col]
                    # Normalize weights for the group to avoid sum(w)=0 issues?
                    # Or just sum(x*w)/sum(w)
                    denom = w.sum()
                    if denom == 0: return np.nan
                    return (x * w).sum() / denom

                # GroupBy apply is slow. Better way:
                # Multiply all agg cols by weight
                # GroupBy sum
                # Divide by sum of weights
                
                # Create weighted columns
                weighted_df = df[group_cols].copy()
                sum_weights = df.groupby(group_cols)[w_col].sum().reset_index()
                
                for c in agg_cols:
                    if c == w_col: continue # Don't weight the weight itself usually? Or do we? 
                    # Usually population is summed, not averaged.
                    # Logic: if variable is extensive (like GDP), sum. If intensive (like Temp, Yield), weighted mean.
                    # This is tricky to infer automatically.
                    # For now, strict weighted mean for everything except the weight col itself (which we sum).
                    
                    term = df[c] * df[w_col]
                    grouped_term = df.groupby(group_cols).apply(lambda x: (x[c] * x[w_col]).sum())
                    # Optimize:
                    # weighted_cols = df[agg_cols].multiply(df[w_col], axis=0)
                    # output = weighted_cols.groupby(df[group_cols]).sum() ...
                    pass
                
                # Simple approach for now to be robust: Use standard averaging but just print warning
                # Implementing full weighted average logic correctly:
                
                grouped = df.groupby(group_cols)
                result = pd.DataFrame()
                
                # Sum weights per group
                w_sums = grouped[w_col].sum()
                
                for c in agg_cols:
                    if c == w_col:
                        # Weights are summed
                        result[c] = w_sums
                    else:
                        # Calculation: sum(val * w) / sum(w)
                        # We can do: (val * w).groupby().sum() / w_sums
                        weighted_series = df[c] * df[w_col]
                        # Need to align the grouping
                        numerator = weighted_series.groupby([df[g] for g in group_cols]).sum()
                        # Align indices
                        result[c] = numerator / w_sums
                
                # Result index is MultiIndex. Reset.
                df = result.reset_index()
                
            else:
                df = df.groupby(group_cols)[agg_cols].mean().reset_index()
                
        elif agg.method == "sum":
            df = df.groupby(group_cols)[agg_cols].sum().reset_index()
            
        return df
