"""
Example: Using Vanna with Databricks and Custom AI Gateway Base URL

This example demonstrates how to:
1. Create a custom OpenAI client with a custom base URL for your AI gateway
2. Integrate Databricks SQL warehouse with Vanna
3. Use ChromaDB for vector storage
4. Train and query the system
"""

import os
from typing import Optional, Dict, Any
from openai import OpenAI
from ..openai.openai_chat import OpenAI_Chat
from ..chromadb.chromadb_vector import ChromaDB_VectorStore

# Import our custom Databricks integration
from databricks import Databricks_SQL


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat, Databricks_SQL):
  """
  Custom Vanna class combining:
  - ChromaDB for vector storage
  - OpenAI Chat with custom base URL for AI gateway
  - Databricks SQL for database operations
  """

  def __init__(self, config: Optional[Dict[str, Any]] = None):
    # Initialize ChromaDB first
    ChromaDB_VectorStore.__init__(self, config=config)

    # Create custom OpenAI client with your AI gateway base URL
    custom_openai_client = OpenAI(
      api_key=config.get('openai_api_key', os.getenv('OPENAI_API_KEY')),
      base_url=config.get('ai_gateway_base_url', os.getenv('AI_GATEWAY_BASE_URL')),
      # You can add other parameters like default_headers for authentication
      default_headers=config.get('ai_gateway_headers', {})
    )

    # Initialize OpenAI Chat with custom client
    OpenAI_Chat.__init__(self, client=custom_openai_client, config=config)

    # Initialize Databricks connection
    Databricks_SQL.__init__(self, config=config)


def main():
  """
  Example usage of Vanna with Databricks and custom AI gateway
  """

  # Configuration for your setup
  config = {
    # OpenAI/AI Gateway settings
    'openai_api_key': 'your-api-key',  # Your API key
    'ai_gateway_base_url': 'https://your-ai-gateway.company.com/v1',  # Your AI gateway URL
    'model': 'gpt-4',  # Model to use
    'ai_gateway_headers': {
      # Any additional headers your AI gateway requires
      'X-Custom-Auth': 'your-auth-token',
      'X-Organization-ID': 'your-org-id'
    },

    # Databricks settings
    'server_hostname': 'your-databricks-workspace.cloud.databricks.com',
    'http_path': '/sql/1.0/warehouses/your-warehouse-id',
    'access_token': 'your-databricks-token',
    'catalog': 'your_catalog',  # Optional: Unity Catalog name
    'schema': 'your_schema',    # Optional: Schema name

    # ChromaDB settings (optional)
    'path': './chroma_db',  # Local path for ChromaDB storage
  }

  # Alternative: Use environment variables instead of config dict
  # Set these environment variables:
  # - OPENAI_API_KEY
  # - AI_GATEWAY_BASE_URL
  # - DATABRICKS_SERVER_HOSTNAME
  # - DATABRICKS_HTTP_PATH
  # - DATABRICKS_TOKEN
  # - DATABRICKS_CATALOG (optional)
  # - DATABRICKS_SCHEMA (optional)

  # Initialize Vanna with custom configuration
  vn = MyVanna(config=config)

  # Connect to Databricks
  print("Connecting to Databricks...")
  vn.connect_to_db()

  # Get database schema for training
  print("\nGetting database schema...")
  schema_info = vn.get_database_schema()
  for table_name, columns in schema_info.items():
    print(f"Table: {table_name}")
    for col in columns[:5]:  # Show first 5 columns
      print(f"  - {col}")
    if len(columns) > 5:
      print(f"  ... and {len(columns) - 5} more columns")
    print()

  # Training Phase
  print("=== TRAINING VANNA ===")

  # 1. Train on DDL (table structures)
  print("Training on table structures...")
  for table_name in list(schema_info.keys())[:5]:  # Train on first 5 tables
    ddl = vn.get_table_definition(table_name)
    if ddl:
      vn.train(ddl=ddl)
      print(f"✓ Trained on {table_name}")

  # 2. Train on documentation
  print("\nTraining on business documentation...")
  documentation = """
    Business Rules:
    - Revenue is calculated as sum of order_amount * quantity
    - Active customers are those with orders in the last 90 days
    - Product categories include: Electronics, Clothing, Books, Home, Sports
    - Order status can be: pending, processing, shipped, delivered, cancelled
    """
  vn.train(documentation=documentation)
  print("✓ Trained on business documentation")

  # 3. Train on sample SQL queries (optional)
  print("\nTraining on sample SQL queries...")
  sample_queries = [
    "SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id",
    "SELECT product_name, SUM(quantity) as total_sold FROM order_items oi JOIN products p ON oi.product_id = p.id GROUP BY product_name",
    "SELECT DATE_TRUNC('month', order_date) as month, SUM(total_amount) as monthly_revenue FROM orders WHERE order_status = 'delivered' GROUP BY month ORDER BY month"
  ]

  for sql in sample_queries:
    vn.train(sql=sql)
    print(f"✓ Trained on sample query")

  # Query Phase
  print("\n=== QUERYING VANNA ===")

  # Ask natural language questions
  questions = [
    "What are the top 10 customers by total order value?",
    "Show me monthly revenue trends for the last year",
    "Which products are selling the best this quarter?",
    "How many active customers do we have?",
    "What's the average order value by product category?"
  ]

  for question in questions:
    print(f"\n🤔 Question: {question}")
    try:
      # Generate and run SQL
      result = vn.ask(question)

      if result is not None:
        print("✅ Query successful!")
        print("First few results:")
        print(result.head())
      else:
        print("❌ No results returned")

    except Exception as e:
      print(f"❌ Error: {str(e)}")

  # Close connection
  vn.close()
  print("\n🔚 Done!")


def setup_with_environment_variables():
  """
  Alternative setup using environment variables only
  """

  # Verify required environment variables
  required_env_vars = [
    'OPENAI_API_KEY',
    'AI_GATEWAY_BASE_URL',
    'DATABRICKS_SERVER_HOSTNAME',
    'DATABRICKS_HTTP_PATH',
    'DATABRICKS_TOKEN'
  ]

  missing_vars = [var for var in required_env_vars if not os.getenv(var)]
  if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

  # Create Vanna instance with minimal config
  vn = MyVanna(config={
    'model': 'gpt-4',  # You can also set this via environment variable
  })

  return vn


def advanced_ai_gateway_example():
  """
  Example with advanced AI gateway configuration
  """

  config = {
    'openai_api_key': 'your-api-key',
    'ai_gateway_base_url': 'https://gateway.company.com/openai/v1',
    'model': 'gpt-4',
    'ai_gateway_headers': {
      # Custom authentication headers for your AI gateway
      'Authorization': 'Bearer your-gateway-token',
      'X-API-Version': '2024-01-01',
      'X-Request-Source': 'vanna-sql-generator',
      'X-User-ID': 'sql-analyst-001'
    },

    # Databricks configuration
    'server_hostname': os.getenv('DATABRICKS_SERVER_HOSTNAME'),
    'http_path': os.getenv('DATABRICKS_HTTP_PATH'),
    'access_token': os.getenv('DATABRICKS_TOKEN'),
    'catalog': 'production',
    'schema': 'analytics',

    # Session configuration for Databricks
    'session_configuration': {
      'spark.sql.adaptive.enabled': 'true',
      'spark.sql.adaptive.coalescePartitions.enabled': 'true'
    }
  }

  vn = MyVanna(config=config)
  return vn


if __name__ == "__main__":
  main()
