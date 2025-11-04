# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, explode, explode_outer, collect_list, collect_set, 
                                   concat_ws, lit, count, countDistinct, struct, array_join,
                                   when, size, coalesce, first, sum as spark_sum, avg, md5, concat)
from pyspark.sql.types import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime

# COMMAND ----------
# SECTION 1: COMPREHENSIVE DATA EXTRACTION
# COMMAND ----------

# Read the JSON file
df = spark.read.option("multiline", "true").json("/path/to/your/file.json")

# COMMAND ----------
# Extract Datasource Instances with connection details
datasource_instances = df.select(
    explode_outer("datasourceInstances").alias("ds")
).select(
    col("ds.datasourceId").alias("datasource_id"),
    col("ds.datasourceType").alias("datasource_type"),
    col("ds.connectionDetails.server").alias("server"),
    col("ds.connectionDetails.database").alias("database"),
    col("ds.gatewayId").alias("gateway_id")
)

# Create connection string for matching
datasource_instances = datasource_instances.withColumn(
    "connection_string",
    concat_ws(":", col("datasource_type"), col("server"), col("database"))
)

print("Datasource Instances extracted:")
display(datasource_instances)

# COMMAND ----------
# Flatten Workspaces
workspaces_df = df.select(explode("workspaces").alias("workspace"))

workspaces_flat = workspaces_df.select(
    col("workspace.id").alias("workspace_id"),
    col("workspace.name").alias("workspace_name"),
    col("workspace.type").alias("workspace_type"),
    col("workspace.state").alias("workspace_state"),
    col("workspace.reports").alias("reports"),
    col("workspace.datasets").alias("datasets")
)

# COMMAND ----------
# Flatten Reports with all metadata - handle empty reports array
reports_df = workspaces_flat.select(
    col("workspace_id"),
    col("workspace_name"),
    explode_outer("reports").alias("report")
).select(
    col("workspace_id"),
    col("workspace_name"),
    col("report.id").alias("report_id"),
    col("report.name").alias("report_name"),
    col("report.datasetId").alias("dataset_id"),
    col("report.reportType").alias("report_type"),
    col("report.createdDateTime").alias("created_date"),
    col("report.modifiedDateTime").alias("modified_date"),
    col("report.createdBy").alias("created_by"),
    col("report.modifiedBy").alias("modified_by")
).filter(col("report_id").isNotNull())  # Filter out empty report entries

report_count = reports_df.count()
print(f"Total Reports extracted: {report_count}")

if report_count == 0:
    print("WARNING: No reports found in the data. Please check your JSON structure.")
    dbutils.notebook.exit("No reports found to analyze")

# COMMAND ----------
# Flatten Datasets with datasource usage - handle empty datasets array
datasets_df = workspaces_flat.select(
    col("workspace_id"),
    col("workspace_name"),
    explode_outer("datasets").alias("dataset")
).select(
    col("workspace_id"),
    col("workspace_name"),
    col("dataset.id").alias("dataset_id"),
    col("dataset.name").alias("dataset_name"),
    col("dataset.contentProviderType").alias("content_provider_type"),
    col("dataset.targetStorageMode").alias("storage_mode"),
    col("dataset.tables").alias("tables"),
    col("dataset.datasourceUsages").alias("datasource_usages")
).filter(col("dataset_id").isNotNull())  # Filter out empty dataset entries

dataset_count = datasets_df.count()
print(f"Total Datasets extracted: {dataset_count}")

if dataset_count == 0:
    print("WARNING: No datasets found in the data.")

# COMMAND ----------
# Extract datasource usage per dataset - handle empty datasource_usages
dataset_datasources = datasets_df.select(
    col("dataset_id"),
    col("dataset_name"),
    explode_outer("datasource_usages").alias("ds_usage")
).select(
    col("dataset_id"),
    col("dataset_name"),
    col("ds_usage.datasourceInstanceId").alias("datasource_id")
).filter(col("datasource_id").isNotNull())  # Filter out null datasource IDs

# Only join if we have datasource instances
if datasource_instances.count() > 0 and dataset_datasources.count() > 0:
    dataset_datasources = dataset_datasources.join(datasource_instances, "datasource_id", "left")
    
    # Aggregate datasources per dataset
    dataset_datasources_agg = dataset_datasources.groupBy("dataset_id", "dataset_name").agg(
        collect_set("datasource_type").alias("datasource_types"),
        collect_set("server").alias("servers"),
        collect_set("database").alias("databases"),
        collect_set("connection_string").alias("connection_strings"),
        count("datasource_id").alias("datasource_count")
    )
    
    print("Dataset to Datasource mapping:")
    display(dataset_datasources_agg.limit(10))
else:
    # Create empty dataframe with schema if no datasources
    print("WARNING: No datasource instances found. Creating empty datasource mapping.")
    schema = StructType([
        StructField("dataset_id", StringType(), True),
        StructField("dataset_name", StringType(), True),
        StructField("datasource_types", ArrayType(StringType()), True),
        StructField("servers", ArrayType(StringType()), True),
        StructField("databases", ArrayType(StringType()), True),
        StructField("connection_strings", ArrayType(StringType()), True),
        StructField("datasource_count", LongType(), True)
    ])
    dataset_datasources_agg = spark.createDataFrame([], schema)

# COMMAND ----------
# Flatten Tables with source information - handle empty tables array
tables_df = datasets_df.select(
    col("workspace_id"),
    col("dataset_id"),
    col("dataset_name"),
    explode_outer("tables").alias("table")
).select(
    col("workspace_id"),
    col("dataset_id"),
    col("dataset_name"),
    col("table.name").alias("table_name"),
    col("table.isHidden").alias("table_is_hidden"),
    col("table.storageMode").alias("table_storage_mode"),
    col("table.columns").alias("columns"),
    col("table.measures").alias("measures"),
    col("table.source").alias("source")
).filter(col("table_name").isNotNull())  # Filter out empty table entries

tables_count = tables_df.count()
print(f"Total Tables extracted: {tables_count}")

if tables_count == 0:
    print("WARNING: No tables found in datasets.")

# COMMAND ----------
# Flatten Columns with full details - handle empty columns array
columns_df = tables_df.select(
    col("workspace_id"),
    col("dataset_id"),
    col("dataset_name"),
    col("table_name"),
    explode_outer("columns").alias("column")
).select(
    col("workspace_id"),
    col("dataset_id"),
    col("dataset_name"),
    col("table_name"),
    col("column.name").alias("column_name"),
    col("column.dataType").alias("data_type"),
    col("column.columnType").alias("column_type"),
    col("column.isHidden").alias("is_hidden"),
    col("column.expression").alias("column_expression")
).filter(col("column_name").isNotNull())  # Filter out empty column entries

# Separate calculated columns
calculated_columns = columns_df.filter(col("column_type") == "Calculated")

columns_count = columns_df.count()
calc_columns_count = calculated_columns.count()
print(f"Total Columns: {columns_count}")
print(f"Calculated Columns: {calc_columns_count}")

if columns_count == 0:
    print("WARNING: No columns found in tables.")

# COMMAND ----------
# Flatten Measures with full expressions - handle empty measures array
measures_df = tables_df.select(
    col("workspace_id"),
    col("dataset_id"),
    col("dataset_name"),
    col("table_name"),
    explode_outer("measures").alias("measure")
).select(
    col("workspace_id"),
    col("dataset_id"),
    col("dataset_name"),
    col("table_name"),
    col("measure.name").alias("measure_name"),
    col("measure.expression").alias("measure_expression"),
    col("measure.isHidden").alias("measure_is_hidden")
).filter(col("measure_name").isNotNull())  # Filter out empty measure entries

measures_count = measures_df.count()
print(f"Total Measures: {measures_count}")

if measures_count > 0:
    display(measures_df.limit(10))
else:
    print("WARNING: No measures found in tables.")

# COMMAND ----------
# SECTION 2: CREATE COMPREHENSIVE FEATURE AGGREGATIONS
# COMMAND ----------

# Aggregate columns per dataset - handle empty dataframes
if columns_df.count() > 0:
    columns_agg = columns_df.groupBy("dataset_id", "dataset_name").agg(
        collect_list("column_name").alias("column_names"),
        collect_list("data_type").alias("data_types"),
        collect_set("data_type").alias("unique_data_types"),
        count("column_name").alias("total_columns"),
        spark_sum(when(col("column_type") == "Calculated", 1).otherwise(0)).alias("calculated_columns_count"),
        spark_sum(when(col("is_hidden") == True, 1).otherwise(0)).alias("hidden_columns_count"),
        concat_ws(" ", collect_list("column_name")).alias("columns_text")
    )
    
    # Create column name hash for exact schema matching
    columns_agg = columns_agg.withColumn(
        "column_schema_hash",
        md5(array_join(col("column_names"), "|"))
    )
else:
    print("WARNING: No columns to aggregate. Creating empty columns aggregation.")
    schema = StructType([
        StructField("dataset_id", StringType(), True),
        StructField("dataset_name", StringType(), True),
        StructField("column_names", ArrayType(StringType()), True),
        StructField("data_types", ArrayType(StringType()), True),
        StructField("unique_data_types", ArrayType(StringType()), True),
        StructField("total_columns", LongType(), True),
        StructField("calculated_columns_count", LongType(), True),
        StructField("hidden_columns_count", LongType(), True),
        StructField("columns_text", StringType(), True),
        StructField("column_schema_hash", StringType(), True)
    ])
    columns_agg = spark.createDataFrame([], schema)

# COMMAND ----------
# Aggregate measures per dataset with expressions - handle empty dataframes
if measures_df.count() > 0:
    measures_agg = measures_df.groupBy("dataset_id", "dataset_name").agg(
        collect_list("measure_name").alias("measure_names"),
        collect_list("measure_expression").alias("measure_expressions"),
        count("measure_name").alias("total_measures"),
        spark_sum(when(col("measure_is_hidden") == True, 1).otherwise(0)).alias("hidden_measures_count"),
        concat_ws(" ", collect_list("measure_name")).alias("measures_text"),
        concat_ws(" ", collect_list("measure_expression")).alias("expressions_text")
    )
    
    # Create measure signature for similarity
    measures_agg = measures_agg.withColumn(
        "measure_schema_hash",
        md5(array_join(col("measure_names"), "|"))
    )
else:
    print("WARNING: No measures to aggregate. Creating empty measures aggregation.")
    schema = StructType([
        StructField("dataset_id", StringType(), True),
        StructField("dataset_name", StringType(), True),
        StructField("measure_names", ArrayType(StringType()), True),
        StructField("measure_expressions", ArrayType(StringType()), True),
        StructField("total_measures", LongType(), True),
        StructField("hidden_measures_count", LongType(), True),
        StructField("measures_text", StringType(), True),
        StructField("expressions_text", StringType(), True),
        StructField("measure_schema_hash", StringType(), True)
    ])
    measures_agg = spark.createDataFrame([], schema)

# COMMAND ----------
# Aggregate tables per dataset - handle empty dataframes
if tables_df.count() > 0:
    tables_agg = tables_df.select("dataset_id", "dataset_name", "table_name", "table_storage_mode").distinct() \
        .groupBy("dataset_id", "dataset_name").agg(
            collect_list("table_name").alias("table_names"),
            count("table_name").alias("total_tables"),
            concat_ws(" ", collect_list("table_name")).alias("tables_text")
        )
else:
    print("WARNING: No tables to aggregate. Creating empty tables aggregation.")
    schema = StructType([
        StructField("dataset_id", StringType(), True),
        StructField("dataset_name", StringType(), True),
        StructField("table_names", ArrayType(StringType()), True),
        StructField("total_tables", LongType(), True),
        StructField("tables_text", StringType(), True)
    ])
    tables_agg = spark.createDataFrame([], schema)

# COMMAND ----------
# SECTION 3: CREATE MASTER FEATURE TABLE
# COMMAND ----------

# Join all features together
dataset_features = columns_agg \
    .join(measures_agg, ["dataset_id", "dataset_name"], "left") \
    .join(tables_agg, ["dataset_id", "dataset_name"], "left") \
    .join(dataset_datasources_agg, ["dataset_id", "dataset_name"], "left")

# Add reports information
report_features = reports_df.join(dataset_features, "dataset_id", "left")

# Fill nulls for aggregation columns
report_features = report_features.fillna({
    "total_columns": 0,
    "total_measures": 0,
    "total_tables": 0,
    "datasource_count": 0,
    "calculated_columns_count": 0,
    "hidden_columns_count": 0,
    "hidden_measures_count": 0,
    "columns_text": "",
    "measures_text": "",
    "expressions_text": "",
    "tables_text": ""
})

# Create comprehensive feature text for TF-IDF
report_features = report_features.withColumn(
    "feature_text",
    concat_ws(" ",
        col("report_name"),
        col("dataset_name"),
        col("tables_text"),
        col("columns_text"),
        col("measures_text"),
        col("expressions_text")
    )
)

# Create metadata text for additional context
report_features = report_features.withColumn(
    "metadata_text",
    concat_ws(" ",
        array_join(coalesce(col("datasource_types"), array()), " "),
        array_join(coalesce(col("servers"), array()), " "),
        array_join(coalesce(col("databases"), array()), " ")
    )
)

print(f"Master feature table created with {report_features.count()} reports")
display(report_features.select("report_id", "report_name", "total_columns", "total_measures", 
                                "total_tables", "datasource_count").limit(10))

# COMMAND ----------
# Cache the feature table for performance with 15k reports
report_features.cache()
report_features.count()  # Force cache

# COMMAND ----------
# SECTION 4: ADVANCED SIMILARITY CALCULATION
# COMMAND ----------

# Convert to Pandas for sklearn processing
pdf = report_features.select(
    "report_id",
    "report_name", 
    "dataset_id",
    "dataset_name",
    "workspace_name",
    "workspace_id",
    "total_columns",
    "total_measures",
    "total_tables",
    "calculated_columns_count",
    "datasource_count",
    "column_schema_hash",
    "measure_schema_hash",
    "feature_text",
    "metadata_text",
    "created_by",
    "modified_by",
    "created_date",
    "modified_date"
).toPandas()

pdf = pdf.fillna('')

print(f"Processing {len(pdf)} reports for similarity analysis...")

# COMMAND ----------
# Multi-level similarity calculation

# 1. TF-IDF on feature text (columns, measures, tables, expressions)
print("Calculating TF-IDF similarity on features...")
try:
    tfidf_feature = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 3),
        min_df=min(2, pdf_count),  # Adjust min_df based on dataset size
        max_df=0.8,
        sublinear_tf=True
    )
    tfidf_feature_matrix = tfidf_feature.fit_transform(pdf['feature_text'])
    feature_similarity = cosine_similarity(tfidf_feature_matrix)
    print("✓ Feature similarity calculated")
except Exception as e:
    print(f"WARNING: Could not calculate feature similarity: {e}")
    feature_similarity = np.ones((pdf_count, pdf_count))

# 2. TF-IDF on metadata (datasources, servers)
print("Calculating TF-IDF similarity on metadata...")
try:
    tfidf_meta = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=1
    )
    tfidf_meta_matrix = tfidf_meta.fit_transform(pdf['metadata_text'])
    metadata_similarity = cosine_similarity(tfidf_meta_matrix)
    print("✓ Metadata similarity calculated")
except Exception as e:
    print(f"WARNING: Could not calculate metadata similarity: {e}")
    metadata_similarity = np.ones((pdf_count, pdf_count))

# 3. Numerical feature similarity (columns count, measures count, etc.)
print("Calculating numerical feature similarity...")
try:
    numerical_features = pdf[['total_columns', 'total_measures', 'total_tables', 
                              'calculated_columns_count', 'datasource_count']].values
    
    # Check if we have any variation in numerical features
    if np.std(numerical_features) > 0:
        # Normalize numerical features
        scaler = MinMaxScaler()
        numerical_features_scaled = scaler.fit_transform(numerical_features)
        
        # Calculate Euclidean distance and convert to similarity
        from scipy.spatial.distance import cdist
        euclidean_dist = cdist(numerical_features_scaled, numerical_features_scaled, 'euclidean')
        max_dist = np.max(euclidean_dist)
        numerical_similarity = 1 - (euclidean_dist / max_dist) if max_dist > 0 else np.ones_like(euclidean_dist)
    else:
        print("WARNING: No variation in numerical features, using uniform similarity")
        numerical_similarity = np.ones((pdf_count, pdf_count))
    print("✓ Numerical similarity calculated")
except Exception as e:
    print(f"WARNING: Could not calculate numerical similarity: {e}")
    numerical_similarity = np.ones((pdf_count, pdf_count))

# 4. Exact schema match (column names and measure names)
print("Calculating exact schema matches...")
try:
    exact_match_similarity = np.zeros((pdf_count, pdf_count))
    for i in range(pdf_count):
        for j in range(i, pdf_count):
            col_hash_i = pdf.iloc[i]['column_schema_hash']
            col_hash_j = pdf.iloc[j]['column_schema_hash']
            measure_hash_i = pdf.iloc[i]['measure_schema_hash']
            measure_hash_j = pdf.iloc[j]['measure_schema_hash']
            
            # Handle empty hashes
            col_match = 1.0 if (col_hash_i and col_hash_j and col_hash_i == col_hash_j) else 0.0
            measure_match = 1.0 if (measure_hash_i and measure_hash_j and measure_hash_i == measure_hash_j) else 0.0
            
            exact_match_similarity[i][j] = (col_match + measure_match) / 2
            exact_match_similarity[j][i] = exact_match_similarity[i][j]
    print("✓ Exact schema matching calculated")
except Exception as e:
    print(f"WARNING: Could not calculate exact schema matches: {e}")
    exact_match_similarity = np.zeros((pdf_count, pdf_count))

# COMMAND ----------
# Weighted combined similarity
# Adjust weights based on your priorities
FEATURE_WEIGHT = 0.40      # Column names, measure names, expressions
METADATA_WEIGHT = 0.15     # Datasources, servers
NUMERICAL_WEIGHT = 0.20    # Counts of columns, measures, tables
EXACT_MATCH_WEIGHT = 0.25  # Exact schema matches

combined_similarity = (
    FEATURE_WEIGHT * feature_similarity +
    METADATA_WEIGHT * metadata_similarity +
    NUMERICAL_WEIGHT * numerical_similarity +
    EXACT_MATCH_WEIGHT * exact_match_similarity
)

print("Combined similarity matrix calculated!")

# COMMAND ----------
# SECTION 5: GENERATE CONSOLIDATION REPORT
# COMMAND ----------

def generate_consolidation_report(min_similarity=0.75, max_results_per_report=10):
    """
    Generate comprehensive consolidation report for similar reports
    """
    if pdf_count < 2:
        print("Need at least 2 reports to generate consolidation report")
        return pd.DataFrame()
    
    consolidation_candidates = []
    processed_pairs = set()
    
    for i in range(pdf_count):
        # Get similarity scores for this report
        sim_scores = combined_similarity[i]
        
        # Find similar reports (excluding itself)
        similar_indices = np.where((sim_scores >= min_similarity) & (np.arange(len(sim_scores)) != i))[0]
        
        if len(similar_indices) > 0:
            # Sort by similarity
            similar_indices = similar_indices[np.argsort(-sim_scores[similar_indices])][:max_results_per_report]
            
            for j in similar_indices:
                # Create unique pair identifier to avoid duplicates
                pair_id = tuple(sorted([i, j]))
                
                if pair_id not in processed_pairs:
                    processed_pairs.add(pair_id)
                    
                    # Calculate detailed similarity breakdown
                    consolidation_candidates.append({
                        'report1_id': pdf.iloc[i]['report_id'],
                        'report1_name': pdf.iloc[i]['report_name'],
                        'report1_workspace': pdf.iloc[i]['workspace_name'],
                        'report1_dataset': pdf.iloc[i]['dataset_name'],
                        'report1_columns': int(pdf.iloc[i]['total_columns']) if pd.notna(pdf.iloc[i]['total_columns']) else 0,
                        'report1_measures': int(pdf.iloc[i]['total_measures']) if pd.notna(pdf.iloc[i]['total_measures']) else 0,
                        'report1_tables': int(pdf.iloc[i]['total_tables']) if pd.notna(pdf.iloc[i]['total_tables']) else 0,
                        'report1_created_by': pdf.iloc[i]['created_by'] if pd.notna(pdf.iloc[i]['created_by']) else '',
                        'report1_modified_date': str(pdf.iloc[i]['modified_date']) if pd.notna(pdf.iloc[i]['modified_date']) else '',
                        
                        'report2_id': pdf.iloc[j]['report_id'],
                        'report2_name': pdf.iloc[j]['report_name'],
                        'report2_workspace': pdf.iloc[j]['workspace_name'],
                        'report2_dataset': pdf.iloc[j]['dataset_name'],
                        'report2_columns': int(pdf.iloc[j]['total_columns']) if pd.notna(pdf.iloc[j]['total_columns']) else 0,
                        'report2_measures': int(pdf.iloc[j]['total_measures']) if pd.notna(pdf.iloc[j]['total_measures']) else 0,
                        'report2_tables': int(pdf.iloc[j]['total_tables']) if pd.notna(pdf.iloc[j]['total_tables']) else 0,
                        'report2_created_by': pdf.iloc[j]['created_by'] if pd.notna(pdf.iloc[j]['created_by']) else '',
                        'report2_modified_date': str(pdf.iloc[j]['modified_date']) if pd.notna(pdf.iloc[j]['modified_date']) else '',
                        
                        'overall_similarity': round(float(combined_similarity[i][j]) * 100, 2),
                        'feature_similarity': round(float(feature_similarity[i][j]) * 100, 2),
                        'metadata_similarity': round(float(metadata_similarity[i][j]) * 100, 2),
                        'numerical_similarity': round(float(numerical_similarity[i][j]) * 100, 2),
                        'exact_match_score': round(float(exact_match_similarity[i][j]) * 100, 2),
                        
                        'same_workspace': pdf.iloc[i]['workspace_name'] == pdf.iloc[j]['workspace_name'],
                        'same_dataset': pdf.iloc[i]['dataset_id'] == pdf.iloc[j]['dataset_id'],
                        'column_diff': abs(int(pdf.iloc[i]['total_columns']) - int(pdf.iloc[j]['total_columns'])),
                        'measure_diff': abs(int(pdf.iloc[i]['total_measures']) - int(pdf.iloc[j]['total_measures'])),
                        
                        'consolidation_priority': 'HIGH' if combined_similarity[i][j] >= 0.90 else 
                                                 'MEDIUM' if combined_similarity[i][j] >= 0.80 else 'LOW'
                    })
    
    return pd.DataFrame(consolidation_candidates)

# COMMAND ----------
# Generate consolidation report with 75% similarity threshold
print("Generating consolidation report...")
consolidation_report = generate_consolidation_report(min_similarity=0.75, max_results_per_report=10)

if len(consolidation_report) > 0:
    print(f"\nConsolidation Report Summary:")
    print(f"Total similar report pairs found: {len(consolidation_report)}")
    print(f"HIGH priority pairs (>90% similar): {len(consolidation_report[consolidation_report['consolidation_priority']=='HIGH'])}")
    print(f"MEDIUM priority pairs (80-90% similar): {len(consolidation_report[consolidation_report['consolidation_priority']=='MEDIUM'])}")
    print(f"LOW priority pairs (75-80% similar): {len(consolidation_report[consolidation_report['consolidation_priority']=='LOW'])}")
else:
    print("\nNo similar report pairs found with 75% similarity threshold.")
    print("Try lowering the similarity threshold or verify your data has enough variation.")

# COMMAND ----------
# Display top consolidation candidates
if len(consolidation_report) > 0:
    top_candidates = consolidation_report.sort_values('overall_similarity', ascending=False).head(50)
    print("\nTop 50 Consolidation Candidates:")
    display(top_candidates[[
        'report1_name', 'report1_workspace', 'report1_dataset',
        'report2_name', 'report2_workspace', 'report2_dataset',
        'overall_similarity', 'consolidation_priority', 
        'same_workspace', 'same_dataset'
    ]])
else:
    print("\nNo consolidation candidates to display.")

# COMMAND ----------
# SECTION 6: IDENTIFY REPORT CLUSTERS FOR CONSOLIDATION
# COMMAND ----------

from collections import defaultdict

def identify_report_clusters(min_similarity=0.80):
    """
    Identify clusters of similar reports that can be consolidated together
    """
    # Build adjacency list
    graph = defaultdict(list)
    
    for i in range(len(pdf)):
        for j in range(i+1, len(pdf)):
            if combined_similarity[i][j] >= min_similarity:
                graph[i].append(j)
                graph[j].append(i)
    
    # Find connected components (clusters)
    visited = set()
    clusters = []
    
    def dfs(node, cluster):
        visited.add(node)
        cluster.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, cluster)
    
    for i in range(len(pdf)):
        if i not in visited and i in graph:
            cluster = []
            dfs(i, cluster)
            if len(cluster) > 1:
                clusters.append(cluster)
    
    # Create cluster report
    cluster_report = []
    for idx, cluster in enumerate(clusters):
        # Calculate average similarity within cluster
        similarities = []
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                similarities.append(combined_similarity[cluster[i]][cluster[j]])
        
        avg_similarity = np.mean(similarities) if similarities else 0
        
        cluster_report.append({
            'cluster_id': idx + 1,
            'cluster_size': len(cluster),
            'avg_similarity': round(float(avg_similarity) * 100, 2),
            'report_names': [pdf.iloc[i]['report_name'] for i in cluster],
            'report_ids': [pdf.iloc[i]['report_id'] for i in cluster],
            'workspaces': list(set([pdf.iloc[i]['workspace_name'] for i in cluster])),
            'datasets': list(set([pdf.iloc[i]['dataset_name'] for i in cluster])),
            'potential_savings': f"{len(cluster) - 1} reports can be consolidated"
        })
    
    return pd.DataFrame(cluster_report)

# COMMAND ----------
# Generate cluster report
print("Identifying report clusters...")
cluster_report = identify_report_clusters(min_similarity=0.80)

print(f"\nCluster Analysis Summary:")
print(f"Total clusters found: {len(cluster_report)}")
print(f"Total reports that can be consolidated: {cluster_report['cluster_size'].sum() - len(cluster_report)}")

# Sort by cluster size
cluster_report_sorted = cluster_report.sort_values('cluster_size', ascending=False)
print("\nTop 20 Largest Clusters:")
display(cluster_report_sorted.head(20))

# COMMAND ----------
# SECTION 7: DETAILED SIMILARITY ANALYSIS
# COMMAND ----------

def analyze_report_pair(report1_name, report2_name):
    """
    Detailed analysis of two specific reports
    """
    r1_idx = pdf[pdf['report_name'] == report1_name].index
    r2_idx = pdf[pdf['report_name'] == report2_name].index
    
    if len(r1_idx) == 0 or len(r2_idx) == 0:
        print("One or both reports not found!")
        return None
    
    i, j = r1_idx[0], r2_idx[0]
    
    # Get detailed feature comparison
    r1_features = report_features.filter(col("report_name") == report1_name).first()
    r2_features = report_features.filter(col("report_name") == report2_name).first()
    
    analysis = {
        'report1': report1_name,
        'report2': report2_name,
        'overall_similarity': f"{combined_similarity[i][j]*100:.2f}%",
        'feature_similarity': f"{feature_similarity[i][j]*100:.2f}%",
        'metadata_similarity': f"{metadata_similarity[i][j]*100:.2f}%",
        'numerical_similarity': f"{numerical_similarity[i][j]*100:.2f}%",
        'exact_match_score': f"{exact_match_similarity[i][j]*100:.2f}%",
    }
    
    # Compare columns
    cols1 = set(r1_features['column_names'] if r1_features and r1_features['column_names'] else [])
    cols2 = set(r2_features['column_names'] if r2_features and r2_features['column_names'] else [])
    
    if cols1 or cols2:
        analysis['shared_columns'] = list(cols1.intersection(cols2))
        analysis['unique_to_report1'] = list(cols1 - cols2)
        analysis['unique_to_report2'] = list(cols2 - cols1)
        analysis['column_overlap'] = f"{len(cols1.intersection(cols2))/max(len(cols1.union(cols2)), 1)*100:.1f}%"
    
    # Compare measures
    measures1 = set(r1_features['measure_names'] if r1_features and r1_features['measure_names'] else [])
    measures2 = set(r2_features['measure_names'] if r2_features and r2_features['measure_names'] else [])
    
    if measures1 or measures2:
        analysis['shared_measures'] = list(measures1.intersection(measures2))
        analysis['unique_measures_report1'] = list(measures1 - measures2)
        analysis['unique_measures_report2'] = list(measures2 - measures1)
        analysis['measure_overlap'] = f"{len(measures1.intersection(measures2))/max(len(measures1.union(measures2)), 1)*100:.1f}%"
    
    return analysis

# COMMAND ----------
# SECTION 8: EXPORT RESULTS
# COMMAND ----------

# Convert to Spark DataFrames for export
consolidation_spark = spark.createDataFrame(consolidation_report)
cluster_spark = spark.createDataFrame(cluster_report)

# Save to Delta tables
consolidation_spark.write.format("delta").mode("overwrite").saveAsTable("powerbi_analytics.report_consolidation_candidates")
cluster_spark.write.format("delta").mode("overwrite").saveAsTable("powerbi_analytics.report_clusters")

# Save detailed feature table
report_features.write.format("delta").mode("overwrite").saveAsTable("powerbi_analytics.report_features")

print("Results saved to Delta tables!")

# COMMAND ----------
# SECTION 9: SUMMARY STATISTICS AND INSIGHTS
# COMMAND ----------

# Overall statistics
print("\n" + "="*80)
print("CONSOLIDATION OPPORTUNITY SUMMARY")
print("="*80)

total_reports = len(pdf)
reports_with_duplicates = len(set(consolidation_report['report1_id'].tolist() + consolidation_report['report2_id'].tolist()))
potential_savings = len(cluster_report['cluster_size'].sum() - len(cluster_report))

print(f"\nTotal Reports Analyzed: {total_reports}")
print(f"Reports with Similar Duplicates: {reports_with_duplicates} ({reports_with_duplicates/total_reports*100:.1f}%)")
print(f"Potential Reports to Consolidate: {potential_savings}")
print(f"Similar Report Pairs Found: {len(consolidation_report)}")
print(f"Report Clusters Identified: {len(cluster_report)}")

# Priority breakdown
high_priority = consolidation_report[consolidation_report['consolidation_priority'] == 'HIGH']
medium_priority = consolidation_report[consolidation_report['consolidation_priority'] == 'MEDIUM']
low_priority = consolidation_report[consolidation_report['consolidation_priority'] == 'LOW']

print(f"\nConsolidation Priority Breakdown:")
print(f"  HIGH Priority (>90% similar): {len(high_priority)} pairs")
print(f"  MEDIUM Priority (80-90% similar): {len(medium_priority)} pairs")
print(f"  LOW Priority (75-80% similar): {len(low_priority)} pairs")

# Same workspace/dataset analysis
same_workspace = consolidation_report[consolidation_report['same_workspace'] == True]
same_dataset = consolidation_report[consolidation_report['same_dataset'] == True]

print(f"\nCross-Workspace/Dataset Analysis:")
print(f"  Same Workspace: {len(same_workspace)} pairs ({len(same_workspace)/len(consolidation_report)*100:.1f}%)")
print(f"  Same Dataset: {len(same_dataset)} pairs ({len(same_dataset)/len(consolidation_report)*100:.1f}%)")

# COMMAND ----------
# SECTION 10: EXPORT TO EXCEL FOR STAKEHOLDER REVIEW
# COMMAND ----------

# Create Excel report with multiple sheets
excel_path = "/dbfs/tmp/report_consolidation_analysis.xlsx"

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Sheet 1: Executive Summary
    summary_data = {
        'Metric': [
            'Total Reports Analyzed',
            'Reports with Similar Duplicates',
            'Potential Reports to Consolidate',
            'Similar Pairs Found',
            'Report Clusters Identified',
            'High Priority Pairs (>90%)',
            'Medium Priority Pairs (80-90%)',
            'Low Priority Pairs (75-80%)'
        ],
        'Value': [
            total_reports,
            reports_with_duplicates,
            potential_savings,
            len(consolidation_report),
            len(cluster_report),
            len(high_priority),
            len(medium_priority),
            len(low_priority)
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive Summary', index=False)
    
    # Sheet 2: High Priority Consolidations
    high_priority.to_excel(writer, sheet_name='High Priority', index=False)
    
    # Sheet 3: All Consolidation Candidates
    consolidation_report.to_excel(writer, sheet_name='All Candidates', index=False)
    
    # Sheet 4: Report Clusters
    cluster_report_sorted.to_excel(writer, sheet_name='Report Clusters', index=False)
    
    # Sheet 5: Report Metadata
    pdf[['report_name', 'workspace_name', 'dataset_name', 'total_columns', 
         'total_measures', 'total_tables', 'created_by', 'modified_date']].to_excel(
        writer, sheet_name='Report Metadata', index=False)

print(f"\nExcel report generated: {excel_path}")
print("Download it using: dbutils.fs.cp('file:/dbfs/tmp/report_consolidation_analysis.xlsx', 'dbfs:/FileStore/report_consolidation_analysis.xlsx')")

# COMMAND ----------
# SECTION 11: INTERACTIVE DASHBOARD DATA PREPARATION
# COMMAND ----------

# Prepare data for visualization
viz_data = consolidation_report.copy()

# Add categorization
viz_data['similarity_category'] = pd.cut(
    viz_data['overall_similarity'], 
    bins=[0, 80, 90, 100], 
    labels=['75-80%', '80-90%', '90-100%']
)

# Workspace comparison
viz_data['workspace_comparison'] = viz_data.apply(
    lambda x: 'Same Workspace' if x['same_workspace'] else 'Cross Workspace', axis=1
)

# Dataset comparison
viz_data['dataset_comparison'] = viz_data.apply(
    lambda x: 'Same Dataset' if x['same_dataset'] else 'Different Dataset', axis=1
)

# Summary by workspace
workspace_summary = viz_data.groupby('report1_workspace').agg({
    'report1_id': 'count',
    'overall_similarity': 'mean'
}).reset_index()
workspace_summary.columns = ['workspace', 'duplicate_pairs', 'avg_similarity']
workspace_summary = workspace_summary.sort_values('duplicate_pairs', ascending=False)

print("\nWorkspaces with Most Duplicate Reports:")
display(workspace_summary.head(20))

# COMMAND ----------
# SECTION 12: ADVANCED FILTERING AND SEARCH FUNCTIONS
# COMMAND ----------

def find_all_similar_reports(report_name, min_similarity=0.75):
    """
    Find all reports similar to a given report
    """
    if report_name not in pdf['report_name'].values:
        print(f"Report '{report_name}' not found!")
        return None
    
    idx = pdf[pdf['report_name'] == report_name].index[0]
    sim_scores = combined_similarity[idx]
    
    # Get all similar reports
    similar_indices = np.where((sim_scores >= min_similarity) & (np.arange(len(sim_scores)) != idx))[0]
    similar_indices = similar_indices[np.argsort(-sim_scores[similar_indices])]
    
    results = pdf.iloc[similar_indices][['report_id', 'report_name', 'workspace_name', 
                                          'dataset_name', 'total_columns', 'total_measures', 
                                          'total_tables', 'created_by', 'modified_date']].copy()
    
    results['overall_similarity'] = [f"{sim_scores[i]*100:.2f}%" for i in similar_indices]
    results['feature_similarity'] = [f"{feature_similarity[idx][i]*100:.2f}%" for i in similar_indices]
    results['metadata_similarity'] = [f"{metadata_similarity[idx][i]*100:.2f}%" for i in similar_indices]
    
    return results

def find_duplicates_by_workspace(workspace_name, min_similarity=0.75):
    """
    Find all duplicate reports within a specific workspace
    """
    workspace_reports = pdf[pdf['workspace_name'] == workspace_name]
    
    if workspace_reports.empty:
        print(f"Workspace '{workspace_name}' not found!")
        return None
    
    duplicates = consolidation_report[
        ((consolidation_report['report1_workspace'] == workspace_name) |
         (consolidation_report['report2_workspace'] == workspace_name)) &
        (consolidation_report['overall_similarity'] >= min_similarity * 100)
    ]
    
    return duplicates.sort_values('overall_similarity', ascending=False)

def find_duplicates_by_dataset(dataset_name, min_similarity=0.75):
    """
    Find all duplicate reports using a specific dataset
    """
    duplicates = consolidation_report[
        ((consolidation_report['report1_dataset'] == dataset_name) |
         (consolidation_report['report2_dataset'] == dataset_name)) &
        (consolidation_report['overall_similarity'] >= min_similarity * 100)
    ]
    
    return duplicates.sort_values('overall_similarity', ascending=False)

def get_consolidation_recommendations(cluster_id):
    """
    Get detailed consolidation recommendations for a specific cluster
    """
    cluster_info = cluster_report[cluster_report['cluster_id'] == cluster_id]
    
    if cluster_info.empty:
        print(f"Cluster {cluster_id} not found!")
        return None
    
    cluster_data = cluster_info.iloc[0]
    report_ids = cluster_data['report_ids']
    
    # Get full details for all reports in cluster
    cluster_reports = pdf[pdf['report_id'].isin(report_ids)].copy()
    
    # Add recommendation logic
    cluster_reports['recommendation'] = ''
    
    # Find the most recently modified report
    cluster_reports['modified_date_dt'] = pd.to_datetime(cluster_reports['modified_date'], errors='coerce')
    most_recent_idx = cluster_reports['modified_date_dt'].idxmax()
    
    if pd.notna(most_recent_idx):
        cluster_reports.loc[most_recent_idx, 'recommendation'] = 'KEEP - Most Recently Updated'
        cluster_reports.loc[cluster_reports.index != most_recent_idx, 'recommendation'] = 'CONSIDER DEPRECATING'
    
    return cluster_reports[['report_name', 'workspace_name', 'dataset_name', 'total_columns', 
                            'total_measures', 'created_by', 'modified_date', 'recommendation']]

# COMMAND ----------
# SECTION 13: CREATE INTERACTIVE WIDGETS FOR EXPLORATION
# COMMAND ----------

# Create widgets for interactive exploration
dbutils.widgets.dropdown("analysis_type", "Find Similar Reports", 
                        ["Find Similar Reports", "Workspace Analysis", "Dataset Analysis", "Cluster Details"],
                        "Analysis Type:")

dbutils.widgets.text("search_term", "", "Search Term (Report/Workspace/Dataset Name):")
dbutils.widgets.dropdown("min_similarity", "75", ["70", "75", "80", "85", "90", "95"], "Minimum Similarity %:")

analysis_type = dbutils.widgets.get("analysis_type")
search_term = dbutils.widgets.get("search_term")
min_sim = float(dbutils.widgets.get("min_similarity")) / 100

if search_term:
    print(f"\n{'='*80}")
    print(f"Analysis: {analysis_type}")
    print(f"Search Term: {search_term}")
    print(f"Minimum Similarity: {min_sim*100}%")
    print(f"{'='*80}\n")
    
    if analysis_type == "Find Similar Reports":
        results = find_all_similar_reports(search_term, min_sim)
        if results is not None and not results.empty:
            print(f"Found {len(results)} similar reports:")
            display(results)
        elif results is not None:
            print("No similar reports found with the specified threshold.")
    
    elif analysis_type == "Workspace Analysis":
        results = find_duplicates_by_workspace(search_term, min_sim)
        if results is not None and not results.empty:
            print(f"Found {len(results)} duplicate pairs in workspace '{search_term}':")
            display(results)
        elif results is not None:
            print(f"No duplicates found in workspace '{search_term}' with the specified threshold.")
    
    elif analysis_type == "Dataset Analysis":
        results = find_duplicates_by_dataset(search_term, min_sim)
        if results is not None and not results.empty:
            print(f"Found {len(results)} duplicate pairs using dataset '{search_term}':")
            display(results)
        elif results is not None:
            print(f"No duplicates found for dataset '{search_term}' with the specified threshold.")
    
    elif analysis_type == "Cluster Details":
        try:
            cluster_id = int(search_term)
            results = get_consolidation_recommendations(cluster_id)
            if results is not None:
                print(f"Consolidation recommendations for Cluster {cluster_id}:")
                display(results)
        except ValueError:
            print("For Cluster Details, please enter a numeric cluster ID")

# COMMAND ----------
# SECTION 14: ACTIONABLE INSIGHTS AND RECOMMENDATIONS
# COMMAND ----------

def generate_action_plan():
    """
    Generate prioritized action plan for report consolidation
    """
    action_items = []
    
    # HIGH PRIORITY: Exact duplicates (>95% similar)
    exact_duplicates = consolidation_report[consolidation_report['overall_similarity'] >= 95]
    if len(exact_duplicates) > 0:
        action_items.append({
            'priority': 'CRITICAL',
            'action': 'Review Exact Duplicates',
            'count': len(exact_duplicates),
            'description': 'Reports with >95% similarity - likely complete duplicates',
            'estimated_savings': f"{len(exact_duplicates)} reports",
            'next_steps': 'Review with report owners and deprecate redundant copies'
        })
    
    # HIGH PRIORITY: Same dataset, same workspace, high similarity
    same_ws_ds = consolidation_report[
        (consolidation_report['same_workspace'] == True) &
        (consolidation_report['same_dataset'] == True) &
        (consolidation_report['overall_similarity'] >= 85)
    ]
    if len(same_ws_ds) > 0:
        action_items.append({
            'priority': 'HIGH',
            'action': 'Consolidate Same-Workspace Reports',
            'count': len(same_ws_ds),
            'description': 'Similar reports in same workspace using same dataset',
            'estimated_savings': f"{len(same_ws_ds)} reports",
            'next_steps': 'Merge reports or create parameterized version'
        })
    
    # MEDIUM PRIORITY: Large clusters
    large_clusters = cluster_report[cluster_report['cluster_size'] >= 5]
    if len(large_clusters) > 0:
        action_items.append({
            'priority': 'MEDIUM',
            'action': 'Consolidate Large Report Clusters',
            'count': len(large_clusters),
            'description': f'Clusters with 5+ similar reports',
            'estimated_savings': f"{large_clusters['cluster_size'].sum() - len(large_clusters)} reports",
            'next_steps': 'Create standardized template report for each cluster'
        })
    
    # MEDIUM PRIORITY: Cross-workspace duplicates
    cross_workspace = consolidation_report[
        (consolidation_report['same_workspace'] == False) &
        (consolidation_report['overall_similarity'] >= 80)
    ]
    if len(cross_workspace) > 0:
        action_items.append({
            'priority': 'MEDIUM',
            'action': 'Standardize Cross-Workspace Reports',
            'count': len(cross_workspace),
            'description': 'Similar reports across different workspaces',
            'estimated_savings': f"{len(cross_workspace)} reports",
            'next_steps': 'Create shared workspace with common reports'
        })
    
    return pd.DataFrame(action_items)

action_plan = generate_action_plan()
print("\n" + "="*80)
print("PRIORITIZED ACTION PLAN")
print("="*80 + "\n")
display(action_plan)

# COMMAND ----------
# SECTION 15: GENERATE STAKEHOLDER COMMUNICATION TEMPLATES
# COMMAND ----------

def generate_stakeholder_email(cluster_id):
    """
    Generate email template for stakeholder communication about specific cluster
    """
    cluster_info = cluster_report[cluster_report['cluster_id'] == cluster_id]
    
    if cluster_info.empty:
        return None
    
    cluster_data = cluster_info.iloc[0]
    report_ids = cluster_data['report_ids']
    cluster_reports = pdf[pdf['report_id'].isin(report_ids)]
    
    email_template = f"""
Subject: Power BI Report Consolidation Opportunity - Cluster {cluster_id}

Dear Stakeholders,

Our automated analysis has identified a consolidation opportunity for {cluster_data['cluster_size']} similar Power BI reports:

REPORTS IN THIS CLUSTER:
"""
    
    for idx, report in cluster_reports.iterrows():
        email_template += f"\n  • {report['report_name']}"
        email_template += f"\n    Workspace: {report['workspace_name']}"
        email_template += f"\n    Dataset: {report['dataset_name']}"
        email_template += f"\n    Owner: {report['created_by']}\n"
    
    email_template += f"""
SIMILARITY ANALYSIS:
  • Average Similarity Score: {cluster_data['avg_similarity']}%
  • Potential Savings: {cluster_data['potential_savings']}

RECOMMENDATION:
We recommend reviewing these reports to determine if they can be consolidated into a single 
standardized report or if parameters can be used to reduce redundancy.

NEXT STEPS:
1. Review each report's usage and audience
2. Identify unique requirements vs. common patterns
3. Create consolidated report design
4. Plan migration and deprecation timeline

Please reply with your availability for a consolidation planning session.

Best regards,
Analytics Team

---
This email was auto-generated by the Power BI Consolidation Analysis system.
Report ID: Cluster-{cluster_id}
Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
"""
    
    return email_template

# Example: Generate email for largest cluster
if len(cluster_report) > 0:
    largest_cluster_id = cluster_report.sort_values('cluster_size', ascending=False).iloc[0]['cluster_id']
    sample_email = generate_stakeholder_email(int(largest_cluster_id))
    print("\n" + "="*80)
    print("SAMPLE STAKEHOLDER EMAIL")
    print("="*80)
    print(sample_email)

# COMMAND ----------
# SECTION 16: MONITORING AND TRACKING
# COMMAND ----------

# Create tracking table for consolidation progress
tracking_data = []

for idx, row in consolidation_report.iterrows():
    tracking_data.append({
        'pair_id': f"{row['report1_id'][:8]}_{row['report2_id'][:8]}",
        'report1_name': row['report1_name'],
        'report2_name': row['report2_name'],
        'similarity_score': row['overall_similarity'],
        'priority': row['consolidation_priority'],
        'status': 'IDENTIFIED',
        'assigned_to': '',
        'target_date': '',
        'actual_date': '',
        'notes': '',
        'identified_date': datetime.now().strftime('%Y-%m-%d')
    })

tracking_df = pd.DataFrame(tracking_data)
tracking_spark = spark.createDataFrame(tracking_df)

# Save tracking table
tracking_spark.write.format("delta").mode("overwrite").saveAsTable("powerbi_analytics.consolidation_tracking")

print("\nConsolidation tracking table created!")
print("Use this table to track progress on consolidation efforts.")

# COMMAND ----------
# SECTION 17: PERFORMANCE METRICS
# COMMAND ----------

print("\n" + "="*80)
print("ANALYSIS PERFORMANCE METRICS")
print("="*80 + "\n")

print(f"Reports Processed: {len(pdf):,}")
print(f"Similarity Calculations: {len(pdf) * (len(pdf) - 1) // 2:,}")
print(f"Features Analyzed per Report:")
print(f"  - Average Columns: {pdf['total_columns'].mean():.1f}")
print(f"  - Average Measures: {pdf['total_measures'].mean():.1f}")
print(f"  - Average Tables: {pdf['total_tables'].mean():.1f}")
print(f"\nResults Generated:")
print(f"  - Consolidation Candidates: {len(consolidation_report):,}")
print(f"  - Report Clusters: {len(cluster_report):,}")
print(f"  - Delta Tables Created: 4")
print(f"  - Excel Report: Generated")

# COMMAND ----------
# SECTION 18: QUICK REFERENCE QUERIES
# COMMAND ----------

print("\n" + "="*80)
print("QUICK REFERENCE - SQL QUERIES FOR ANALYSIS")
print("="*80 + "\n")

queries = """
-- Find all high priority consolidation candidates
SELECT * FROM powerbi_analytics.report_consolidation_candidates
WHERE consolidation_priority = 'HIGH'
ORDER BY overall_similarity DESC;

-- Find largest clusters
SELECT cluster_id, cluster_size, avg_similarity, potential_savings
FROM powerbi_analytics.report_clusters
ORDER BY cluster_size DESC
LIMIT 20;

-- Find reports by similarity threshold
SELECT report1_name, report2_name, overall_similarity, 
       feature_similarity, metadata_similarity
FROM powerbi_analytics.report_consolidation_candidates
WHERE overall_similarity >= 90
ORDER BY overall_similarity DESC;

-- Track consolidation progress
SELECT status, priority, COUNT(*) as count
FROM powerbi_analytics.consolidation_tracking
GROUP BY status, priority
ORDER BY priority, status;

-- Find duplicates in specific workspace
SELECT * FROM powerbi_analytics.report_consolidation_candidates
WHERE report1_workspace = 'YOUR_WORKSPACE_NAME'
   OR report2_workspace = 'YOUR_WORKSPACE_NAME'
ORDER BY overall_similarity DESC;
"""

print(queries)

# COMMAND ----------
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80 + "\n")
print("✓ All data processed and saved to Delta tables")
print("✓ Consolidation report generated")
print("✓ Excel export created")
print("✓ Interactive widgets configured")
print("\nNext Steps:")
print("1. Review the consolidation_report for high-priority candidates")
print("2. Use the interactive widgets to explore specific reports/workspaces")
print("3. Download the Excel report for stakeholder review")
print("4. Use SQL queries to further analyze the results")
print("5. Begin consolidation planning with report owners")