# Databricks Integration for Vanna AI

This guide shows how to add Databricks support to Vanna AI and configure it with a custom AI gateway base URL.

## Installation

1. **Install required dependencies:**

```bash
pip install vanna
pip install databricks-sql-connector
pip install databricks-sdk
pip install pandas
pip install openai
```

2. **Add the Databricks integration files to your Vanna installation:**

Place the `databricks_integration.py` file in the appropriate directory:
```
vanna/
├── src/vanna/
│   ├── databricks/
│   │   ├── __init__.py
│   │   └── databricks_sql.py  # <- Place the integration code here
```

## Configuration

### Environment Variables

Set up the following environment variables:

```bash
# OpenAI/AI Gateway settings
export OPENAI_API_KEY="your-openai-api-key"
export AI_GATEWAY_BASE_URL="https://your-ai-gateway.company.com/v1"

# Databricks settings
export DATABRICKS_SERVER_HOSTNAME="your-workspace.cloud.databricks.com"
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/your-warehouse-id"
export DATABRICKS_TOKEN="your-databricks-personal-access-token"
export DATABRICKS_CATALOG="your_catalog"  # Optional
export DATABRICKS_SCHEMA="your_schema"    # Optional
```

### Getting Databricks Connection Details

1. **Server Hostname**: Your Databricks workspace URL without `https://`
    - Example: `your-workspace.cloud.databricks.com`

2. **HTTP Path**: Found in your SQL Warehouse details
    - Go to SQL Warehouses → Your Warehouse → Connection details
    - Example: `/sql/1.0/warehouses/abc123def456`

3. **Access Token**: Create a personal access token
    - User Settings → Access tokens → Generate new token

## Usage Examples

### Basic Usage

```python
from vanna_databricks_example import MyVanna

# Configuration
config = {
    'openai_api_key': 'your-key',
    'ai_gateway_base_url': 'https://gateway.company.com/v1',
    'model': 'gpt-4',
    'server_hostname': 'workspace.cloud.databricks.com',
    'http_path': '/sql/1.0/warehouses/warehouse-id',
    'access_token': 'databricks-token'
}

# Initialize
vn = MyVanna(config=config)

# Connect and train
vn.connect_to_db()
vn.train(ddl="CREATE TABLE customers (id INT, name VARCHAR(100), email VARCHAR(100))")

# Ask questions
result = vn.ask("How many customers do we have?")
print(result)
```

### Custom AI Gateway Configuration

For corporate environments with AI gateways:

```python
config = {
    'ai_gateway_base_url': 'https://internal-ai-proxy.company.com/openai/v1',
    'ai_gateway_headers': {
        'Authorization': 'Bearer your-internal-token',
        'X-Organization': 'your-org-id',
        'X-Project': 'analytics-project'
    },
    # ... other config
}
```

### Unity Catalog Support

For Unity Catalog enabled workspaces:

```python
config = {
    'catalog': 'production',
    'schema': 'analytics', 
    # ... other config
}
```

## File Structure

Add these files to your Vanna fork:

```
src/vanna/databricks/
├── __init__.py
└── databricks_sql.py

examples/
└── databricks_example.py
```

### `src/vanna/databricks/__init__.py`

```python
from .databricks_sql import Databricks_SQL, Databricks_Compute

__all__ = ['Databricks_SQL', 'Databricks_Compute']
```

## Features

### Databricks_SQL Class

- ✅ SQL Warehouse connectivity
- ✅ Unity Catalog support
- ✅ DDL extraction
- ✅ Query execution
- ✅ Schema introspection
- ✅ Session configuration

### AI Gateway Support

- ✅ Custom base URLs
- ✅ Custom headers
- ✅ Authentication tokens
- ✅ Proxy support
- ✅ Load balancing compatible

## Integration Steps for Vanna Repository

### 1. Fork and Clone Vanna Repository

```bash
git clone https://github.com/your-username/vanna.git
cd vanna
```

### 2. Create Databricks Module

Create the directory structure:
```bash
mkdir -p src/vanna/databricks
touch src/vanna/databricks/__init__.py
```

### 3. Add Integration Files

Copy the `Databricks_SQL` and `Databricks_Compute` classes to:
- `src/vanna/databricks/databricks_sql.py`

### 4. Update Main Package

Add to `src/vanna/__init__.py`:
```python
# Add this import
try:
    from .databricks import Databricks_SQL, Databricks_Compute
except ImportError:
    pass  # Optional dependency
```

### 5. Add to Requirements

Create `requirements-databricks.txt`:
```
databricks-sql-connector>=3.0.0
databricks-sdk>=0.20.0
```

Update `setup.py` or `pyproject.toml`:
```python
extras_require = {
    "databricks": [
        "databricks-sql-connector>=3.0.0",
        "databricks-sdk>=0.20.0"
    ]
}
```

## Testing the Integration

### Unit Tests

Create `tests/test_databricks.py`:

```python
import pytest
import os
from unittest.mock import Mock, patch
from vanna.databricks.databricks_sql import Databricks_SQL

class TestDatabricksSQL:
    def test_init_with_config(self):
        config = {
            'server_hostname': 'test.databricks.com',
            'http_path': '/sql/1.0/warehouses/test',
            'access_token': 'test-token'
        }
        db = Databricks_SQL(config=config)
        assert db.server_hostname == 'test.databricks.com'
        assert db.http_path == '/sql/1.0/warehouses/test'
        assert db.access_token == 'test-token'
    
    def test_init_with_env_vars(self):
        with patch.dict(os.environ, {
            'DATABRICKS_SERVER_HOSTNAME': 'env.databricks.com',
            'DATABRICKS_HTTP_PATH': '/sql/1.0/warehouses/env',
            'DATABRICKS_TOKEN': 'env-token'
        }):
            db = Databricks_SQL()
            assert db.server_hostname == 'env.databricks.com'
    
    @patch('databricks.sql.connect')
    def test_connect_to_db(self, mock_connect):
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        config = {
            'server_hostname': 'test.databricks.com',
            'http_path': '/sql/1.0/warehouses/test',
            'access_token': 'test-token',
            'catalog': 'test_catalog',
            'schema': 'test_schema'
        }
        
        db = Databricks_SQL(config=config)
        db.connect_to_db()
        
        mock_connect.assert_called_once()
        mock_cursor.execute.assert_any_call("USE CATALOG test_catalog")
        mock_cursor.execute.assert_any_call("USE SCHEMA test_schema")
```

### Integration Test

Create a test script to verify the complete workflow:

```python
# test_integration.py
import os
from vanna_databricks_example import MyVanna

def test_full_integration():
    """Test the complete integration"""
    
    # Skip if environment variables not set
    required_vars = [
        'DATABRICKS_SERVER_HOSTNAME',
        'DATABRICKS_HTTP_PATH', 
        'DATABRICKS_TOKEN',
        'OPENAI_API_KEY'
    ]
    
    if not all(os.getenv(var) for var in required_vars):
        print("Skipping integration test - missing environment variables")
        return
    
    config = {
        'ai_gateway_base_url': os.getenv('AI_GATEWAY_BASE_URL', 'https://api.openai.com/v1'),
        'model': 'gpt-3.5-turbo'  # Use cheaper model for testing
    }
    
    vn = MyVanna(config=config)
    
    try:
        # Test connection
        vn.connect_to_db()
        print("✅ Database connection successful")
        
        # Test schema retrieval
        schema = vn.get_database_schema()
        print(f"✅ Retrieved schema for {len(schema)} tables")
        
        # Test simple query
        result = vn.run_sql("SELECT 1 as test_column")
        assert result is not None
        print("✅ Test query successful")
        
        print("🎉 All integration tests passed!")
        
    finally:
        vn.close()

if __name__ == "__main__":
    test_full_integration()
```

## Advanced Configuration Examples

### Enterprise AI Gateway with Authentication

```python
config = {
    'ai_gateway_base_url': 'https://ai-gateway.corp.com/api/v1/openai',
    'ai_gateway_headers': {
        'Authorization': 'Bearer company-issued-token',
        'X-API-Key': 'secondary-auth-key',
        'X-Department': 'data-analytics',
        'X-Cost-Center': '12345',
        'User-Agent': 'VannaAI/1.0 (Internal)'
    },
    'openai_api_key': 'sk-proxy-key-from-gateway'
}
```

### Multi-Environment Configuration

```python
def get_config(environment='dev'):
    """Get configuration for different environments"""
    
    base_config = {
        'model': 'gpt-4',
        'ai_gateway_headers': {
            'X-Environment': environment,
            'X-Application': 'vanna-sql-generator'
        }
    }
    
    if environment == 'dev':
        base_config.update({
            'ai_gateway_base_url': 'https://dev-ai-gateway.company.com/v1',
            'server_hostname': 'dev-databricks.company.com',
            'catalog': 'dev_analytics'
        })
    elif environment == 'prod':
        base_config.update({
            'ai_gateway_base_url': 'https://prod-ai-gateway.company.com/v1', 
            'server_hostname': 'prod-databricks.company.com',
            'catalog': 'production_analytics'
        })
    
    return base_config
```

### Performance Optimization

```python
config = {
    # Databricks session optimization
    'session_configuration': {
        'spark.sql.adaptive.enabled': 'true',
        'spark.sql.adaptive.coalescePartitions.enabled': 'true',
        'spark.sql.adaptive.skewJoin.enabled': 'true',
        'spark.databricks.delta.optimizeWrite.enabled': 'true',
        'spark.databricks.delta.autoCompact.enabled': 'true'
    },
    
    # AI Gateway optimization  
    'ai_gateway_headers': {
        'X-Request-Priority': 'high',
        'X-Cache-TTL': '300',  # 5 minutes
        'X-Retry-Policy': 'exponential'
    }
}
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   ```python
   # Verify credentials
   from databricks import sql
   
   connection = sql.connect(
       server_hostname="your-hostname",
       http_path="your-path", 
       access_token="your-token"
   )
   print("Connection successful!")
   ```

2. **AI Gateway Issues**
   ```python
   # Test gateway directly
   from openai import OpenAI
   
   client = OpenAI(
       base_url="https://your-gateway.com/v1",
       api_key="your-key"
   )
   
   response = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": "Hello"}]
   )
   print(response.choices[0].message.content)
   ```

3. **Permission Errors**
    - Ensure Databricks token has SQL warehouse access
    - Check Unity Catalog permissions if using catalogs
    - Verify AI gateway allows your IP/domain

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your Vanna code here
vn = MyVanna(config=config)
```

## Contributing Back to Vanna

To contribute this integration back to the main Vanna repository:

1. **Create a Pull Request** with:
    - Integration code
    - Tests
    - Documentation
    - Example usage

2. **Follow Vanna's contribution guidelines**
    - Code style consistency
    - Comprehensive tests
    - Clear documentation

3. **Update relevant documentation**
    - README.md
    - API documentation
    - Usage examples

This integration provides a robust foundation for using Vanna AI with Databricks while supporting enterprise AI gateway requirements. The modular design allows for easy customization and extension based on your specific needs.