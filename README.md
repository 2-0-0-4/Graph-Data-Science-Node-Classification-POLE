# Graph Machine Learning for Criminal Risk Prediction
 
Node classification on the **POLE-50** dataset using Neo4j Graph Data Science (GDS).
 
## Overview
 
The study evaluates multiple node classification pipelines using:

- Total nodes: 61,521  
- Total relationships: 105,840  
- Node types: Person, Crime, Location, PostCode, Area, Officer, Phone, PhoneCall, Email, Vehicle, Object  
- Person nodes used for classification: 369  
- Criminally involved individuals: 29  

The full methodology and results are documented in [`report/`](report/).
 
## Repository Structure
 
```
├── gds_project.cypher      # All Cypher queries: preprocessing, feature engineering, projections, pipelines
├── pole-50.dump            # Neo4j database dump (restore to load the full graph)
└── report/
    ├── report.pdf          # Final IEEE-format paper
    ├── report.tex          # LaTeX source
    └── schema.png          # Graph schema diagram
```
 
## Dataset
 
The POLE-50 graph contains **61,521 nodes** and **105,840 relationships** across 11 node types (Person, Crime, Location, Vehicle, Phone, etc.). Classification targets are the **369 Person nodes**, of which **29 are criminally involved**.

## Evaluation Metrics

Model performance was evaluated using:

- **Accuracy** 
- **F1-Weighted**
- **F1-Macro** 

## Setup
 
1. Install [Neo4j Desktop](https://neo4j.com/download/) with the **Graph Data Science** plugin.
2. Restore the database dump:
   ```
   neo4j-admin database load --from-path=. pole-50 --overwrite-destination
   ```
3. Start the database and open Neo4j Browser.
4. Run `gds_project.cypher` in order — it covers data quality checks, feature engineering, graph projections, and all pipeline configurations.
