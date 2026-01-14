from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Any, Generator
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataBackend(ABC):
    """
    Abstract base class for data access backends.
    """
    
    @abstractmethod
    def load_data(
        self, 
        columns: Optional[List[str]] = None, 
        filters: Optional[Dict[str, Any]] = None,
        sample_size: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Load data into a pandas DataFrame.
        
        Args:
            columns: List of columns to selected
            filters: Dictionary of filters {col: value} or {col: [values]}
            sample_size: Number of rows to sample (for test mode)
            random_seed: Random seed for sampling
        """
        pass
        
    @abstractmethod
    def get_unique_values(self, column: str) -> List[Any]:
        """Get unique values for a column."""
        pass

class PandasBackend(DataBackend):
    """
    In-memory backend using Pandas. Suitable for country/ADM1 levels.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    @classmethod
    def from_csv(cls, path: str, **kwargs):
        logger.info(f"Loading CSV from {path}")
        return cls(pd.read_csv(path, **kwargs))
        
    @classmethod
    def from_parquet(cls, path: str, **kwargs):
        logger.info(f"Loading Parquet from {path}")
        return cls(pd.read_parquet(path, **kwargs))
        
    def load_data(
        self, 
        columns: Optional[List[str]] = None, 
        filters: Optional[Dict[str, Any]] = None,
        sample_size: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        df = self.df
        
        # Apply filters
        if filters:
            for col, val in filters.items():
                if isinstance(val, (list, tuple)):
                    df = df[df[col].isin(val)]
                else:
                    df = df[df[col] == val]
        
        # Select columns
        if columns:
            # Deduplicate columns requested
            columns = list(dict.fromkeys(columns))
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns not found: {missing}")
            df = df[columns]
            
        # Sampling
        if sample_size and len(df) > sample_size:
            logger.info(f"Sampling {sample_size} rows from {len(df)} total")
            df = df.sample(n=sample_size, random_state=random_seed)
            
        return df.copy()

    def get_unique_values(self, column: str) -> List[Any]:
        return sorted(self.df[column].unique().tolist())

class DuckDBBackend(DataBackend):
    """
    Database backend using DuckDB. Suitable for impact regions / large data.
    """
    def __init__(self, db_path: str, table_name: str, read_only: bool = True):
        import duckdb
        self.db_path = db_path
        self.table_name = table_name
        self.read_only = read_only
        self._con = None
        
    def _connect(self):
        if self._con is None:
            import duckdb
            self._con = duckdb.connect(self.db_path, read_only=self.read_only)
        return self._con
        
    def close(self):
        if self._con:
            self._con.close()
            self._con = None
            
    def load_data(
        self, 
        columns: Optional[List[str]] = None, 
        filters: Optional[Dict[str, Any]] = None,
        sample_size: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        con = self._connect()
        
        # Build query
        if columns:
            columns = list(dict.fromkeys(columns))
        cols_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols_str} FROM {self.table_name}"
        params = []
        
        where_clauses = []
        if filters:
            for col, val in filters.items():
                if isinstance(val, (list, tuple)):
                    placeholders = ", ".join(["?"] * len(val))
                    where_clauses.append(f"{col} IN ({placeholders})")
                    params.extend(val)
                else:
                    where_clauses.append(f"{col} = ?")
                    params.append(val)
                    
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
            
        if sample_size:
            # DuckDB generic sampling (BERNOULLI is approximate percentage)
            # For exact number, simpler to limit if needed, or use reservoir sampling
            # Using ORDER BY random() LIMIT N is expensive but exact
            query += f" USING SAMPLE {sample_size} ROWS (Reservoir)"
            
        logger.debug(f"Executing query: {query}")
        return con.execute(query, params).df()

    def get_unique_values(self, column: str) -> List[Any]:
        con = self._connect()
        query = f"SELECT DISTINCT {column} FROM {self.table_name} ORDER BY {column}"
        return [r[0] for r in con.execute(query).fetchall()]
