"""
Databricks integration for Vanna AI
This module provides integration with Databricks SQL warehouses and compute clusters
"""
import os
import pandas as pd
from typing import Optional, List, Tuple, Any, Dict
from ..base import VannaBase

try:
  from databricks import sql
  from databricks.sdk import WorkspaceClient
  from databricks.sdk.core import Config
except ImportError:
  raise ImportError("Please install databricks-sql-connector and databricks-sdk: pip install databricks-sql-connector databricks-sdk")


class Databricks_SQL(VannaBase):
  """
  Integration with Databricks SQL warehouses for Vanna AI

  This class extends VannaBase to work with Databricks SQL warehouses,
  providing methods to connect, run queries, and retrieve metadata.
  """

  def __init__(self, config: Optional[Dict[str, Any]] = None):
    """
    Initialize Databricks SQL connection

    Args:
        config: Dictionary containing Databricks connection parameters:
            - server_hostname: Databricks workspace hostname
            - http_path: SQL warehouse HTTP path
            - access_token: Personal access token or service principal token
            - catalog: Unity Catalog name (optional)
            - schema: Schema/database name (optional)
            - session_configuration: Additional session config (optional)
    """
    VannaBase.__init__(self, config=config)

    # Get configuration from config dict or environment variables
    self.server_hostname = config.get('server_hostname') if config else None
    self.http_path = config.get('http_path') if config else None
    self.access_token = config.get('access_token') if config else None
    self.catalog = config.get('catalog') if config else None
    self.schema = config.get('schema') if config else None
    self.session_configuration = config.get('session_configuration', {}) if config else {}

    # Fallback to environment variables
    if not self.server_hostname:
      self.server_hostname = os.getenv('DATABRICKS_SERVER_HOSTNAME')
    if not self.http_path:
      self.http_path = os.getenv('DATABRICKS_HTTP_PATH')
    if not self.access_token:
      self.access_token = os.getenv('DATABRICKS_TOKEN')
    if not self.catalog:
      self.catalog = os.getenv('DATABRICKS_CATALOG')
    if not self.schema:
      self.schema = os.getenv('DATABRICKS_SCHEMA', 'default')

    self.connection = None
    self.cursor = None

    # Validate required parameters
    if not all([self.server_hostname, self.http_path, self.access_token]):
      raise ValueError("Missing required Databricks connection parameters. "
                       "Please provide server_hostname, http_path, and access_token")

  def connect_to_db(self) -> None:
    """
    Connect to Databricks SQL warehouse
    """
    try:
      self.connection = sql.connect(
        server_hostname=self.server_hostname,
        http_path=self.http_path,
        access_token=self.access_token,
        session_configuration=self.session_configuration
      )
      self.cursor = self.connection.cursor()

      # Set catalog and schema if provided
      if self.catalog:
        self.cursor.execute(f"USE CATALOG {self.catalog}")
      if self.schema:
        self.cursor.execute(f"USE SCHEMA {self.schema}")

      print(f"Connected to Databricks SQL warehouse")
      if self.catalog:
        print(f"Using catalog: {self.catalog}")
      if self.schema:
        print(f"Using schema: {self.schema}")

    except Exception as e:
      raise ConnectionError(f"Failed to connect to Databricks: {str(e)}")

  def run_sql(self, sql: str) -> Optional[pd.DataFrame]:
    """
    Run SQL query against Databricks

    Args:
        sql: SQL query string

    Returns:
        DataFrame with query results, or None if no results
    """
    if not self.connection or not self.cursor:
      self.connect_to_db()

    try:
      self.cursor.execute(sql)

      # Get column names
      columns = [desc[0] for desc in self.cursor.description] if self.cursor.description else []

      # Fetch results
      rows = self.cursor.fetchall()

      if not rows:
        return None

      # Convert to DataFrame
      df = pd.DataFrame(rows, columns=columns)
      return df

    except Exception as e:
      print(f"Error executing SQL: {str(e)}")
      raise

  def get_table_definition(self, table_name: str) -> Optional[str]:
    """
    Get DDL for a specific table

    Args:
        table_name: Name of the table (can include catalog.schema.table)

    Returns:
        DDL string or None if table not found
    """
    try:
      # Try to get detailed table information
      describe_sql = f"DESCRIBE TABLE EXTENDED {table_name}"
      result = self.run_sql(describe_sql)

      if result is not None and not result.empty:
        # Build DDL from DESCRIBE output
        ddl_parts = [f"-- Table: {table_name}"]

        # Add column definitions
        columns = []
        for _, row in result.iterrows():
          col_name = row.get('col_name', '')
          data_type = row.get('data_type', '')
          comment = row.get('comment', '')

          if col_name and data_type and not col_name.startswith('#'):
            col_def = f"{col_name} {data_type}"
            if comment:
              col_def += f" COMMENT '{comment}'"
            columns.append(col_def)

        if columns:
          ddl_parts.append(f"CREATE TABLE {table_name} (")
          ddl_parts.append(",\n  ".join(f"  {col}" for col in columns))
          ddl_parts.append(")")

        return "\n".join(ddl_parts)

    except Exception as e:
      print(f"Error getting table definition for {table_name}: {str(e)}")

    return None

  def get_related_ddl(self, question: str, **kwargs) -> List[str]:
    """
    Get DDL for tables related to the question

    Args:
        question: Natural language question
        **kwargs: Additional parameters

    Returns:
        List of DDL strings for related tables
    """
    ddl_list = []

    try:
      # Get all tables in current catalog/schema
      if self.catalog and self.schema:
        tables_sql = f"SHOW TABLES IN {self.catalog}.{self.schema}"
      elif self.schema:
        tables_sql = f"SHOW TABLES IN {self.schema}"
      else:
        tables_sql = "SHOW TABLES"

      tables_df = self.run_sql(tables_sql)

      if tables_df is not None and not tables_df.empty:
        # Extract table names
        table_names = []
        if 'tableName' in tables_df.columns:
          table_names = tables_df['tableName'].tolist()
        elif 'table_name' in tables_df.columns:
          table_names = tables_df['table_name'].tolist()

        # Get DDL for each table (limit to avoid too much context)
        for table_name in table_names[:10]:  # Limit to first 10 tables
          full_table_name = table_name
          if self.catalog and self.schema:
            full_table_name = f"{self.catalog}.{self.schema}.{table_name}"
          elif self.schema:
            full_table_name = f"{self.schema}.{table_name}"

          ddl = self.get_table_definition(full_table_name)
          if ddl:
            ddl_list.append(ddl)

    except Exception as e:
      print(f"Error getting related DDL: {str(e)}")

    return ddl_list

  def get_database_schema(self) -> Dict[str, List[str]]:
    """
    Get database schema information

    Returns:
        Dictionary mapping table names to column lists
    """
    schema_info = {}

    try:
      # Get all tables
      if self.catalog and self.schema:
        tables_sql = f"SHOW TABLES IN {self.catalog}.{self.schema}"
      elif self.schema:
        tables_sql = f"SHOW TABLES IN {self.schema}"
      else:
        tables_sql = "SHOW TABLES"

      tables_df = self.run_sql(tables_sql)

      if tables_df is not None and not tables_df.empty:
        table_names = []
        if 'tableName' in tables_df.columns:
          table_names = tables_df['tableName'].tolist()
        elif 'table_name' in tables_df.columns:
          table_names = tables_df['table_name'].tolist()

        for table_name in table_names:
          full_table_name = table_name
          if self.catalog and self.schema:
            full_table_name = f"{self.catalog}.{self.schema}.{table_name}"
          elif self.schema:
            full_table_name = f"{self.schema}.{table_name}"

          # Get columns for this table
          try:
            columns_sql = f"DESCRIBE TABLE {full_table_name}"
            columns_df = self.run_sql(columns_sql)

            if columns_df is not None and not columns_df.empty:
              columns = []
              for _, row in columns_df.iterrows():
                col_name = row.get('col_name', '')
                data_type = row.get('data_type', '')
                if col_name and not col_name.startswith('#'):
                  columns.append(f"{col_name} ({data_type})")

              schema_info[full_table_name] = columns

          except Exception as e:
            print(f"Error getting columns for {full_table_name}: {str(e)}")

    except Exception as e:
      print(f"Error getting database schema: {str(e)}")

    return schema_info

  def close(self) -> None:
    """
    Close Databricks connection
    """
    try:
      if self.cursor:
        self.cursor.close()
      if self.connection:
        self.connection.close()
    except Exception as e:
      print(f"Error closing connection: {str(e)}")

  def __del__(self):
    """
    Cleanup connection when object is destroyed
    """
    self.close()


class Databricks_Compute(VannaBase):
  """
  Integration with Databricks compute clusters using Databricks SDK

  This class provides integration with Databricks all-purpose clusters
  for more advanced analytics workloads.
  """

  def __init__(self, config: Optional[Dict[str, Any]] = None):
    """
    Initialize Databricks compute connection

    Args:
        config: Dictionary containing Databricks connection parameters:
            - host: Databricks workspace URL
            - token: Personal access token
            - cluster_id: Compute cluster ID
            - catalog: Unity Catalog name (optional)
            - schema: Schema name (optional)
    """
    VannaBase.__init__(self, config=config)

    self.host = config.get('host') if config else None
    self.token = config.get('token') if config else None
    self.cluster_id = config.get('cluster_id') if config else None
    self.catalog = config.get('catalog') if config else None
    self.schema = config.get('schema') if config else None

    # Fallback to environment variables
    if not self.host:
      self.host = os.getenv('DATABRICKS_HOST')
    if not self.token:
      self.token = os.getenv('DATABRICKS_TOKEN')
    if not self.cluster_id:
      self.cluster_id = os.getenv('DATABRICKS_CLUSTER_ID')
    if not self.catalog:
      self.catalog = os.getenv('DATABRICKS_CATALOG')
    if not self.schema:
      self.schema = os.getenv('DATABRICKS_SCHEMA', 'default')

    if not all([self.host, self.token, self.cluster_id]):
      raise ValueError("Missing required parameters: host, token, and cluster_id")

    # Initialize Databricks SDK client
    self.client = WorkspaceClient(
      host=self.host,
      token=self.token
    )

  def connect_to_db(self) -> None:
    """
    Verify connection to Databricks workspace and cluster
    """
    try:
      # Check if cluster is accessible
      cluster = self.client.clusters.get(cluster_id=self.cluster_id)
      print(f"Connected to Databricks cluster: {cluster.cluster_name}")
      print(f"Cluster state: {cluster.state}")

      if cluster.state.value != 'RUNNING':
        print("Warning: Cluster is not in RUNNING state")

    except Exception as e:
      raise ConnectionError(f"Failed to connect to Databricks cluster: {str(e)}")

  def run_sql(self, sql: str) -> Optional[pd.DataFrame]:
    """
    Execute SQL using Databricks compute cluster

    Args:
        sql: SQL query string

    Returns:
        DataFrame with results or None
    """
    try:
      # Execute SQL command on cluster
      command_result = self.client.command_execution.execute(
        cluster_id=self.cluster_id,
        language='sql',
        command=sql
      )

      if command_result.status.value == 'Finished' and command_result.results:
        # Parse results (this is a simplified version)
        # In practice, you might need more sophisticated result parsing
        results_text = command_result.results.data

        # For now, return a simple message
        # You would need to implement proper result parsing based on your needs
        print("Query executed successfully on Databricks compute cluster")
        print(f"Results: {results_text}")

        # This is a placeholder - implement actual DataFrame creation
        return pd.DataFrame({'result': [results_text]})

    except Exception as e:
      print(f"Error executing SQL on compute cluster: {str(e)}")
      return None

  def get_related_ddl(self, question: str, **kwargs) -> List[str]:
    """
    Get DDL using compute cluster - placeholder implementation
    """
    # This would be similar to the SQL warehouse implementation
    # but using the compute cluster for execution
    return []
