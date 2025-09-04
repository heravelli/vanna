import pytest
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from vanna.databricks.databricks_sql import Databricks_SQL

class TestDatabricksSQL:

    @pytest.fixture
    def basic_config(self):
        return {
            'server_hostname': 'test.databricks.com',
            'http_path': '/sql/1.0/warehouses/test',
            'access_token': 'test-token'
        }

    @pytest.fixture
    def full_config(self):
        return {
            'server_hostname': 'test.databricks.com',
            'http_path': '/sql/1.0/warehouses/test',
            'access_token': 'test-token',
            'catalog': 'test_catalog',
            'schema': 'test_schema',
            'session_configuration': {'key': 'value'}
        }

    def test_init_with_config(self, basic_config):
        """Test initialization with configuration dictionary"""
        db = Databricks_SQL(config=basic_config)
        assert db.server_hostname == 'test.databricks.com'
        assert db.http_path == '/sql/1.0/warehouses/test'
        assert db.access_token == 'test-token'
        assert db.catalog is None
        assert db.schema == 'default'

    def test_init_with_full_config(self, full_config):
        """Test initialization with full configuration"""
        db = Databricks_SQL(config=full_config)
        assert db.catalog == 'test_catalog'
        assert db.schema == 'test_schema'
        assert db.session_configuration == {'key': 'value'}

    def test_init_with_env_vars(self):
        """Test initialization with environment variables"""
        env_vars = {
            'DATABRICKS_SERVER_HOSTNAME': 'env.databricks.com',
            'DATABRICKS_HTTP_PATH': '/sql/1.0/warehouses/env',
            'DATABRICKS_TOKEN': 'env-token',
            'DATABRICKS_CATALOG': 'env_catalog',
            'DATABRICKS_SCHEMA': 'env_schema'
        }

        with patch.dict(os.environ, env_vars):
            db = Databricks_SQL()
            assert db.server_hostname == 'env.databricks.com'
            assert db.http_path == '/sql/1.0/warehouses/env'
            assert db.access_token == 'env-token'
            assert db.catalog == 'env_catalog'
            assert db.schema == 'env_schema'

    def test_init_missing_required_params(self):
        """Test initialization with missing required parameters"""
        with pytest.raises(ValueError, match="Missing required Databricks connection parameters"):
            Databricks_SQL(config={'server_hostname': 'test.com'})

    @patch('databricks.sql.connect')
    def test_connect_to_db_success(self, mock_connect, full_config):
        """Test successful database connection"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        db = Databricks_SQL(config=full_config)
        db.connect_to_db()

        # Verify connection was established
        mock_connect.assert_called_once_with(
            server_hostname='test.databricks.com',
            http_path='/sql/1.0/warehouses/test',
            access_token='test-token',
            session_configuration={'key': 'value'}
        )

        # Verify catalog and schema were set
        mock_cursor.execute.assert_any_call("USE CATALOG test_catalog")
        mock_cursor.execute.assert_any_call("USE SCHEMA test_schema")

    @patch('databricks.sql.connect')
    def test_connect_to_db_failure(self, mock_connect, basic_config):
        """Test database connection failure"""
        mock_connect.side_effect = Exception("Connection failed")

        db = Databricks_SQL(config=basic_config)
        with pytest.raises(ConnectionError, match="Failed to connect to Databricks"):
            db.connect_to_db()

    def test_run_sql_success(self, basic_config):
        """Test successful SQL execution"""
        db = Databricks_SQL(config=basic_config)

        # Mock connection and cursor
        mock_cursor = Mock()
        mock_cursor.description = [('col1',), ('col2',)]
        mock_cursor.fetchall.return_value = [('row1_col1', 'row1_col2'), ('row2_col1', 'row2_col2')]

        db.cursor = mock_cursor
        db.connection = Mock()

        result = db.run_sql("SELECT * FROM test_table")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['col1', 'col2']
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table")

    def test_run_sql_no_results(self, basic_config):
        """Test SQL execution with no results"""
        db = Databricks_SQL(config=basic_config)

        mock_cursor = Mock()
        mock_cursor.description = None
        mock_cursor.fetchall.return_value = []

        db.cursor = mock_cursor
        db.connection = Mock()

        result = db.run_sql("UPDATE test_table SET col1 = 'value'")

        assert result is None

    @patch.object(Databricks_SQL, 'connect_to_db')
    def test_run_sql_auto_connect(self, mock_connect, basic_config):
        """Test that run_sql auto-connects if not connected"""
        db = Databricks_SQL(config=basic_config)
        db.connection = None
        db.cursor = None

        # After auto-connect, set up mocks
        def setup_connection():
            db.connection = Mock()
            db.cursor = Mock()
            db.cursor.description = [('result',)]
            db.cursor.fetchall.return_value = [(1,)]

        mock_connect.side_effect = setup_connection

        result = db.run_sql("SELECT 1")

        mock_connect.assert_called_once()
        assert result is not None

    def test_get_table_definition_success(self, basic_config):
        """Test successful table definition retrieval"""
        db = Databricks_SQL(config=basic_config)

        # Mock the DESCRIBE TABLE result
        describe_data = {
            'col_name': ['id', 'name', 'email', '# Detailed Table Information', ''],
            'data_type': ['int', 'string', 'string', '', ''],
            'comment': ['Primary key', 'Customer name', 'Email address', '', '']
        }
        mock_df = pd.DataFrame(describe_data)

        with patch.object(db, 'run_sql', return_value=mock_df):
            ddl = db.get_table_definition('test_table')

            assert ddl is not None
            assert 'CREATE TABLE test_table' in ddl
            assert 'id int' in ddl
            assert 'name string' in ddl
            assert 'COMMENT \'Primary key\'' in ddl

    def test_get_table_definition_failure(self, basic_config):
        """Test table definition retrieval failure"""
        db = Databricks_SQL(config=basic_config)

        with patch.object(db, 'run_sql', side_effect=Exception("Table not found")):
            ddl = db.get_table_definition('nonexistent_table')
            assert ddl is None

    def test_get_database_schema_success(self, basic_config):
        """Test successful database schema retrieval"""
        db = Databricks_SQL(config=basic_config)

        # Mock SHOW TABLES result
        tables_data = pd.DataFrame({
            'tableName': ['table1', 'table2']
        })

        # Mock DESCRIBE results for each table
        describe_data = pd.DataFrame({
            'col_name': ['col1', 'col2'],
            'data_type': ['int', 'string'],
            'comment': ['', '']
        })

        with patch.object(db, 'run_sql') as mock_run_sql:
            mock_run_sql.side_effect = [tables_data, describe_data, describe_data]

            schema = db.get_database_schema()

            assert len(schema) == 2
            assert 'table1' in schema
            assert 'table2' in schema
            assert schema['table1'] == ['col1 (int)', 'col2 (string)']

    def test_close_connection(self, basic_config):
        """Test connection cleanup"""
        db = Databricks_SQL(config=basic_config)

        mock_cursor = Mock()
        mock_connection = Mock()
        db.cursor = mock_cursor
        db.connection = mock_connection

        db.close()

        mock_cursor.close.assert_called_once()
        mock_connection.close.assert_called_once()

    def test_close_connection_with_errors(self, basic_config):
        """Test connection cleanup with errors"""
        db = Databricks_SQL(config=basic_config)

        mock_cursor = Mock()
        mock_cursor.close.side_effect = Exception("Close error")
        mock_connection = Mock()

        db.cursor = mock_cursor
        db.connection = mock_connection

        # Should not raise exception
        db.close()

        mock_cursor.close.assert_called_once()
        mock_connection.close.assert_called_once()

class TestDatabricksCompute:
    """Test cases for Databricks_Compute class"""

    def test_init_with_config(self):
        """Test Databricks_Compute initialization"""
        config = {
            'host': 'https://test.databricks.com',
            'token': 'test-token',
            'cluster_id': 'test-cluster-id'
        }

        with patch('databricks.sdk.WorkspaceClient'):
            db = Databricks_SQL(config=config)  # Using SQL class for now
            # Add specific Compute tests when implemented

if __name__ == "__main__":
    pytest.main([__file__])