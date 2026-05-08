// =============================================================================
// SECTION 1: GRAPH PROJECTIONS
// =============================================================================

// Project social graph for betweenness and Louvain
CALL gds.graph.project(
  'social',
  'Person',
  { KNOWS: { orientation: 'UNDIRECTED' } }
);

// Project crime-associates for triangle count
CALL gds.graph.project(
  'crime-associates',
  { Person: { label: 'Person' } },
  { KNOWS: { type: 'KNOWS', orientation: 'UNDIRECTED' } }
)
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount;

// =============================================================================
// SECTION 2: DATA CLEANING
// =============================================================================

// Identify isolated nodes
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n)[0] AS label, count(n) AS isolatedCount
ORDER BY isolatedCount DESC;

// Identify the isolated Person
MATCH (p:Person)
WHERE NOT (p)--()
RETURN p.name + ' ' + p.surname AS person,
       p.nhs_no AS id,
       p.isInvolved AS isInvolved;

// Delete the isolated Person (confirmed not involved)
MATCH (p:Person)
WHERE NOT (p)--()
DELETE p;

// Verify deletion
MATCH (p:Person)
WHERE NOT (p)--()
RETURN count(p) AS remainingIsolatedPersons;

// Check for self-relationships
MATCH (p:Person)-[r]->(p)
RETURN type(r) AS relType, count(r) AS selfLoops;

// Check for null names
MATCH (p:Person)
WHERE p.name IS NULL OR p.surname IS NULL
RETURN count(p) AS personsWithNullNames;

// Check for null NHS numbers
MATCH (p:Person)
WHERE p.nhs_no IS NULL
RETURN count(p) AS personsWithoutId;

// Check betweenness distribution for skew
MATCH (p:Person)
WHERE p.betweenness IS NOT NULL
RETURN
  min(p.betweenness) AS minBetweenness,
  max(p.betweenness) AS maxBetweenness,
  avg(p.betweenness) AS avgBetweenness,
  stDev(p.betweenness) AS stdBetweenness;

// =============================================================================
// SECTION 3: WRITE STRUCTURAL FEATURES VIA GDS
// =============================================================================

// Triangle count — write to in-memory graph then to nodes
CALL gds.triangleCount.mutate('crime-associates', {
  mutateProperty: 'triangleCount'
})
YIELD nodeCount, nodePropertiesWritten;

CALL gds.triangleCount.stream('crime-associates')
YIELD nodeId, triangleCount
WITH gds.util.asNode(nodeId) AS node, triangleCount
SET node.triangleCount = triangleCount;

// Betweenness centrality — write to in-memory graph then to nodes
CALL gds.betweenness.mutate('social', {
  mutateProperty: 'betweenness'
})
YIELD nodePropertiesWritten, mutateMillis;

CALL gds.betweenness.stream('social')
YIELD nodeId, score AS centrality
WITH gds.util.asNode(nodeId) AS node, centrality
SET node.betweenness = centrality;

// Log-transform betweenness to handle right-skewed distribution
// (mean=625, stdDev=879, max=5275 — stdDev exceeds mean)
MATCH (p:Person)
WHERE p.betweenness IS NOT NULL
SET p.betweennessLog = CASE
  WHEN p.betweenness > 0 THEN log(p.betweenness + 1)
  ELSE 0.0
END;

// Louvain community detection — write to in-memory graph then to nodes
CALL gds.louvain.mutate('social', {
  mutateProperty: 'communityId'
})
YIELD nodePropertiesWritten, communityCount, modularity;

CALL gds.louvain.stream('social')
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS node, communityId
SET node.communityId = communityId;

// =============================================================================
// SECTION 4: WRITE LABEL AND BASE FEATURES
// =============================================================================

// Target label — derived from graph traversal not external source
MATCH (p:Person)
SET p.isInvolved = CASE
  WHEN (p)-[:PARTY_TO]->(:Crime) THEN 1
  ELSE 0
END;

// crimeCount — computed but EXCLUDED from ML features (data leakage)
// Both crimeCount and isInvolved derive from PARTY_TO relationships
MATCH (p:Person)
SET p.crimeCount = count{ (p)-[:PARTY_TO]->(:Crime) };

// Degree features
MATCH (p:Person)
SET p.knowsDegree = count{ (p)-[:KNOWS]-() };

MATCH (p:Person)
SET p.familyDegree = count{ (p)-[:FAMILY_REL]-() };

MATCH (p:Person)
SET p.socialDegree = count{ (p)-[:KNOWS_SN]-() };

// =============================================================================
// SECTION 5: WRITE NEIGHBOURHOOD FEATURES
// =============================================================================

// Criminal neighbour count and neighbour crime total
MATCH (p:Person)
OPTIONAL MATCH (p)-[:KNOWS]-(neighbor:Person)-[:PARTY_TO]->(c:Crime)
WITH p,
     count(DISTINCT neighbor) AS criminalNeighborCount,
     count(DISTINCT c) AS neighborCrimeTotal
SET p.criminalNeighborCount = criminalNeighborCount,
    p.neighborCrimeTotal = neighborCrimeTotal;

// Criminal family count
MATCH (p:Person)
OPTIONAL MATCH (p)-[:FAMILY_REL]-(f:Person)-[:PARTY_TO]->(:Crime)
WITH p, count(DISTINCT f) AS linkCount
SET p.criminalFamilyCount = linkCount;

// Criminal social network count
MATCH (p:Person)
OPTIONAL MATCH (p)-[:KNOWS_SN]-(s:Person)-[:PARTY_TO]->(:Crime)
WITH p, count(DISTINCT s) AS linkCount
SET p.criminalSocialCount = linkCount;

// =============================================================================
// SECTION 6: WRITE BEHAVIOURAL FEATURES
// (These showed the highest discriminative power in separability analysis)
// =============================================================================

// Night call ratio — proportion of calls made between midnight and 5am
// Motivated by analysis showing night calls correlate with criminal coordination
MATCH (p:Person)
OPTIONAL MATCH (p)-[:HAS_PHONE]->(ph:Phone)<-[:CALLER]-(pc:PhoneCall)
WITH p,
     count(pc) AS totalCalls,
     sum(CASE WHEN toInteger(split(pc.call_time, ':')[0]) < 5
         THEN 1 ELSE 0 END) AS nightCalls
SET p.nightCallRatio = CASE
  WHEN totalCalls > 0 THEN toFloat(nightCalls) / totalCalls
  ELSE 0.0
END;

// Criminal vehicle count — vehicles linked to crimes the person was party to
MATCH (p:Person)
OPTIONAL MATCH (p)-[:PARTY_TO]->(c:Crime)<-[:INVOLVED_IN]-(v:Vehicle)
WITH p, count(DISTINCT v) AS criminalVehicleCount
SET p.criminalVehicleCount = criminalVehicleCount;

// =============================================================================
// SECTION 7: FEATURE SEPARABILITY ANALYSIS
// Run before pipeline training to verify discriminative value
// =============================================================================

MATCH (p:Person)
RETURN
  p.isInvolved AS label,
  avg(p.knowsDegree) AS avgKnows,
  avg(p.familyDegree) AS avgFamily,
  avg(p.socialDegree) AS avgSocial,
  avg(p.betweenness) AS avgBetweenness,
  avg(p.triangleCount) AS avgTriangles,
  avg(p.criminalNeighborCount) AS avgCrimNeighbors,
  avg(p.criminalFamilyCount) AS avgCrimFamily,
  avg(p.communityId) AS avgCommunity
ORDER BY label;

MATCH (p:Person)
RETURN
  p.isInvolved AS label,
  avg(p.nightCallRatio) AS avgNightCalls,
  avg(p.criminalVehicleCount) AS avgCrimVehicle,
  avg(p.criminalSocialCount) AS avgCrimSocial
ORDER BY label;

// =============================================================================
// SECTION 8: VERIFICATION QUERIES
// =============================================================================

// Verify base features
MATCH (p:Person)
RETURN
  p.name + ' ' + p.surname AS person,
  p.isInvolved AS isInvolved,
  p.crimeCount AS crimeCount,
  p.knowsDegree AS knowsDegree,
  p.familyDegree AS familyDegree,
  p.socialDegree AS socialDegree
ORDER BY p.isInvolved DESC, p.crimeCount DESC
LIMIT 20;

// Verify neighbourhood features
MATCH (p:Person)
RETURN
  p.name + ' ' + p.surname AS person,
  p.isInvolved AS isInvolved,
  p.criminalNeighborCount AS criminalNeighborCount,
  p.criminalFamilyCount AS criminalFamilyCount
ORDER BY p.criminalNeighborCount DESC
LIMIT 20;

// Verify structural features
MATCH (p:Person)
WHERE p.betweenness IS NOT NULL
RETURN
  p.name + ' ' + p.surname AS person,
  p.betweenness AS betweenness,
  p.betweennessLog AS betweennessLog,
  p.triangleCount AS triangleCount,
  p.communityId AS communityId
ORDER BY p.betweenness DESC
LIMIT 20;

// Verify behavioural features
MATCH (p:Person)
RETURN
  p.name + ' ' + p.surname AS person,
  p.isInvolved AS isInvolved,
  p.nightCallRatio AS nightCallRatio,
  p.criminalVehicleCount AS criminalVehicleCount
ORDER BY p.criminalVehicleCount DESC
LIMIT 20;

// Verify isInvolved is strictly binary
MATCH (p:Person)
WHERE p.isInvolved NOT IN [0, 1]
RETURN count(p) AS invalidLabels;

// =============================================================================
// SECTION 9: SOCIAL-FULL PROJECTION (Phase 1 — without embedding)
// =============================================================================

CALL gds.graph.project(
  'social-full',
  {
    Person: {
      properties: [
        'isInvolved',
        'crimeCount',
        'knowsDegree',
        'familyDegree',
        'socialDegree',
        'betweenness',
        'betweennessLog',
        'triangleCount',
        'communityId',
        'criminalNeighborCount',
        'neighborCrimeTotal',
        'criminalFamilyCount',
        'criminalSocialCount',
        'nightCallRatio',
        'criminalVehicleCount'
      ]
    }
  },
  {
    KNOWS: { orientation: 'UNDIRECTED' },
    FAMILY_REL: { orientation: 'UNDIRECTED' },
    KNOWS_SN: { orientation: 'UNDIRECTED' }
  }
)
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount;

// =============================================================================
// SECTION 10: FASTRP EMBEDDINGS — THREE DIMENSIONS TESTED
// =============================================================================

// Generate 32-dim embeddings and write to nodes
CALL gds.fastRP.stream('social-full', {
  embeddingDimension: 32,
  randomSeed: 42
})
YIELD nodeId, embedding
WITH gds.util.asNode(nodeId) AS node, embedding
SET node.embedding32 = embedding;

// Generate 64-dim embeddings and write to nodes (optimal)
CALL gds.fastRP.stream('social-full', {
  embeddingDimension: 64,
  randomSeed: 42
})
YIELD nodeId, embedding
WITH gds.util.asNode(nodeId) AS node, embedding
SET node.embedding = embedding;

// Generate 128-dim embeddings and write to nodes
CALL gds.fastRP.stream('social-full', {
  embeddingDimension: 128,
  randomSeed: 42
})
YIELD nodeId, embedding
WITH gds.util.asNode(nodeId) AS node, embedding
SET node.embedding128 = embedding;

// =============================================================================
// SECTION 11: NODE2VEC EMBEDDINGS
// =============================================================================

// Node2Vec default parameters
CALL gds.node2vec.stream('social-full', {
  embeddingDimension: 64,
  walkLength: 80,
  walksPerNode: 10,
  windowSize: 5,
  negativeSamplingRate: 5,
  randomSeed: 42
})
YIELD nodeId, embedding
WITH gds.util.asNode(nodeId) AS node, embedding
SET node.embeddingN2V = embedding;

// Node2Vec global structure variant (inOutFactor < 1 = explores outward)
CALL gds.node2vec.stream('social-full', {
  embeddingDimension: 64,
  walkLength: 80,
  walksPerNode: 10,
  windowSize: 5,
  returnFactor: 1.0,
  inOutFactor: 0.5,
  negativeSamplingRate: 5,
  randomSeed: 42
})
YIELD nodeId, embedding
WITH gds.util.asNode(nodeId) AS node, embedding
SET node.embeddingN2V_global = embedding;

// Node2Vec local structure variant (inOutFactor > 1 = stays local)
CALL gds.node2vec.stream('social-full', {
  embeddingDimension: 64,
  walkLength: 80,
  walksPerNode: 10,
  windowSize: 5,
  returnFactor: 1.0,
  inOutFactor: 2.0,
  negativeSamplingRate: 5,
  randomSeed: 42
})
YIELD nodeId, embedding
WITH gds.util.asNode(nodeId) AS node, embedding
SET node.embeddingN2V_local = embedding;

// =============================================================================
// SECTION 12: DROP AND RECREATE SOCIAL-FULL WITH ALL EMBEDDINGS
// =============================================================================

CALL gds.graph.drop('social-full');

CALL gds.graph.project(
  'social-full',
  {
    Person: {
      properties: [
        'isInvolved',
        'crimeCount',
        'knowsDegree',
        'familyDegree',
        'socialDegree',
        'betweenness',
        'betweennessLog',
        'triangleCount',
        'communityId',
        'criminalNeighborCount',
        'neighborCrimeTotal',
        'criminalFamilyCount',
        'criminalSocialCount',
        'nightCallRatio',
        'criminalVehicleCount',
        'embedding32',
        'embedding',
        'embedding128',
        'embeddingN2V',
        'embeddingN2V_global',
        'embeddingN2V_local'
      ]
    }
  },
  {
    KNOWS: { orientation: 'UNDIRECTED' },
    FAMILY_REL: { orientation: 'UNDIRECTED' },
    KNOWS_SN: { orientation: 'UNDIRECTED' }
  }
)
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount;

// Verify embeddings
MATCH (p:Person)
WHERE p.embedding IS NOT NULL
RETURN
  p.name + ' ' + p.surname AS person,
  size(p.embedding32) AS dim32,
  size(p.embedding) AS dim64,
  size(p.embedding128) AS dim128,
  size(p.embeddingN2V) AS dimN2V
LIMIT 10;

// =============================================================================
// SECTION 13: PIPELINE A — BEHAVIOURAL FEATURES ONLY
// nightCallRatio + criminalVehicleCount
// Result: F1-Macro 0.6895 (best scalar pipeline)
// =============================================================================

CALL gds.beta.pipeline.nodeClassification.create('pipeline-A');

CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-A', { maxEpochs: 100, penalty: 0.001 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-A', { maxEpochs: 100, penalty: 0.01 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-A', { maxEpochs: 100, penalty: 0.1 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-A', { maxEpochs: 100, penalty: 1.0 });

CALL gds.beta.pipeline.nodeClassification.selectFeatures(
  'pipeline-A',
  ['nightCallRatio', 'criminalVehicleCount']
);

CALL gds.beta.pipeline.nodeClassification.configureSplit('pipeline-A', {
  testFraction: 0.2,
  validationFolds: 5
});

CALL gds.beta.pipeline.nodeClassification.train('social-full', {
  pipeline: 'pipeline-A',
  targetNodeLabels: ['Person'],
  modelName: 'model-A',
  targetProperty: 'isInvolved',
  metrics: ['ACCURACY', 'F1_WEIGHTED', 'F1_MACRO'],
  randomSeed: 42
})
YIELD modelInfo
RETURN
  modelInfo.bestParameters AS bestParams,
  modelInfo.metrics.ACCURACY.test AS testAccuracy,
  modelInfo.metrics.F1_WEIGHTED.test AS testF1Weighted,
  modelInfo.metrics.F1_MACRO.test AS testF1Macro;

// =============================================================================
// SECTION 14: PIPELINE B — BEHAVIOURAL + STRUCTURAL FEATURES
// Adds betweennessLog + triangleCount
// Result: F1-Macro 0.4861 (worse than A — structural features add noise)
// =============================================================================

CALL gds.beta.pipeline.nodeClassification.create('pipeline-B');

CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-B', { maxEpochs: 100, penalty: 0.001 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-B', { maxEpochs: 100, penalty: 0.01 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-B', { maxEpochs: 100, penalty: 0.1 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-B', { maxEpochs: 100, penalty: 1.0 });

CALL gds.beta.pipeline.nodeClassification.selectFeatures(
  'pipeline-B',
  ['nightCallRatio', 'criminalVehicleCount',
   'betweennessLog', 'triangleCount']
);

CALL gds.beta.pipeline.nodeClassification.configureSplit('pipeline-B', {
  testFraction: 0.2,
  validationFolds: 5
});

CALL gds.beta.pipeline.nodeClassification.train('social-full', {
  pipeline: 'pipeline-B',
  targetNodeLabels: ['Person'],
  modelName: 'model-B',
  targetProperty: 'isInvolved',
  metrics: ['ACCURACY', 'F1_WEIGHTED', 'F1_MACRO'],
  randomSeed: 42
})
YIELD modelInfo
RETURN
  modelInfo.bestParameters AS bestParams,
  modelInfo.metrics.ACCURACY.test AS testAccuracy,
  modelInfo.metrics.F1_WEIGHTED.test AS testF1Weighted,
  modelInfo.metrics.F1_MACRO.test AS testF1Macro;

// =============================================================================
// SECTION 15: PIPELINE C — BEHAVIOURAL + NEIGHBOURHOOD FEATURES
// Two variants tested:
// C-full: adds criminalNeighborCount + criminalFamilyCount to Pipeline A
// C-best: uses ONLY the two highest-separating scalar features
// Both result: F1-Macro 0.4861
// This proves failure is due to dataset size not feature quality
// =============================================================================

// C-full variant
CALL gds.beta.pipeline.nodeClassification.create('pipeline-C');

CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-C', { maxEpochs: 100, penalty: 0.001 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-C', { maxEpochs: 100, penalty: 0.01 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-C', { maxEpochs: 100, penalty: 0.1 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-C', { maxEpochs: 100, penalty: 1.0 });

CALL gds.beta.pipeline.nodeClassification.selectFeatures(
  'pipeline-C',
  ['nightCallRatio', 'criminalVehicleCount',
   'criminalNeighborCount', 'criminalFamilyCount']
);

CALL gds.beta.pipeline.nodeClassification.configureSplit('pipeline-C', {
  testFraction: 0.2,
  validationFolds: 5
});

CALL gds.beta.pipeline.nodeClassification.train('social-full', {
  pipeline: 'pipeline-C',
  targetNodeLabels: ['Person'],
  modelName: 'model-C',
  targetProperty: 'isInvolved',
  metrics: ['ACCURACY', 'F1_WEIGHTED', 'F1_MACRO'],
  randomSeed: 42
})
YIELD modelInfo
RETURN
  modelInfo.bestParameters AS bestParams,
  modelInfo.metrics.ACCURACY.test AS testAccuracy,
  modelInfo.metrics.F1_WEIGHTED.test AS testF1Weighted,
  modelInfo.metrics.F1_MACRO.test AS testF1Macro;

// C-best variant — only highest separating features (9.17x and 3.55x)
// Still collapses to 0.4861 — proves zero-inflated distributions fail
// regardless of average separation ratio
CALL gds.beta.pipeline.nodeClassification.create('pipeline-C-best');

CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-C-best', { maxEpochs: 100, penalty: 0.001 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-C-best', { maxEpochs: 100, penalty: 0.01 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-C-best', { maxEpochs: 100, penalty: 0.1 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-C-best', { maxEpochs: 100, penalty: 1.0 });

CALL gds.beta.pipeline.nodeClassification.selectFeatures(
  'pipeline-C-best',
  ['criminalNeighborCount', 'criminalFamilyCount']
);

CALL gds.beta.pipeline.nodeClassification.configureSplit('pipeline-C-best', {
  testFraction: 0.2,
  validationFolds: 5
});

CALL gds.beta.pipeline.nodeClassification.train('social-full', {
  pipeline: 'pipeline-C-best',
  targetNodeLabels: ['Person'],
  modelName: 'model-C-best',
  targetProperty: 'isInvolved',
  metrics: ['ACCURACY', 'F1_WEIGHTED', 'F1_MACRO'],
  randomSeed: 42
})
YIELD modelInfo
RETURN
  modelInfo.bestParameters AS bestParams,
  modelInfo.metrics.ACCURACY.test AS testAccuracy,
  modelInfo.metrics.F1_WEIGHTED.test AS testF1Weighted,
  modelInfo.metrics.F1_MACRO.test AS testF1Macro;

// =============================================================================
// SECTION 16: PIPELINE D — FASTRP EMBEDDINGS (THREE DIMENSIONS)
// D-32: F1-Macro 0.4861 (insufficient dimensions)
// D-64: F1-Macro 0.7751 (optimal — best overall result)
// D-128: F1-Macro 0.70 (overfitting)
// =============================================================================

// Pipeline D-32
CALL gds.beta.pipeline.nodeClassification.create('pipeline-D-32');
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-32', { maxEpochs: 100, penalty: 0.001 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-32', { maxEpochs: 100, penalty: 0.01 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-32', { maxEpochs: 100, penalty: 0.1 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-32', { maxEpochs: 100, penalty: 1.0 });
CALL gds.beta.pipeline.nodeClassification.selectFeatures('pipeline-D-32', ['embedding32']);
CALL gds.beta.pipeline.nodeClassification.configureSplit('pipeline-D-32', { testFraction: 0.2, validationFolds: 5 });

CALL gds.beta.pipeline.nodeClassification.train('social-full', {
  pipeline: 'pipeline-D-32',
  targetNodeLabels: ['Person'],
  modelName: 'model-D-32',
  targetProperty: 'isInvolved',
  metrics: ['ACCURACY', 'F1_WEIGHTED', 'F1_MACRO'],
  randomSeed: 42
})
YIELD modelInfo
RETURN
  modelInfo.bestParameters AS bestParams,
  modelInfo.metrics.ACCURACY.test AS testAccuracy,
  modelInfo.metrics.F1_WEIGHTED.test AS testF1Weighted,
  modelInfo.metrics.F1_MACRO.test AS testF1Macro;

// Pipeline D-64 (optimal)
CALL gds.beta.pipeline.nodeClassification.create('pipeline-D-64');
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-64', { maxEpochs: 100, penalty: 0.001 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-64', { maxEpochs: 100, penalty: 0.01 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-64', { maxEpochs: 100, penalty: 0.1 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-64', { maxEpochs: 100, penalty: 1.0 });
CALL gds.beta.pipeline.nodeClassification.selectFeatures('pipeline-D-64', ['embedding']);
CALL gds.beta.pipeline.nodeClassification.configureSplit('pipeline-D-64', { testFraction: 0.2, validationFolds: 5 });

CALL gds.beta.pipeline.nodeClassification.train('social-full', {
  pipeline: 'pipeline-D-64',
  targetNodeLabels: ['Person'],
  modelName: 'model-D-64',
  targetProperty: 'isInvolved',
  metrics: ['ACCURACY', 'F1_WEIGHTED', 'F1_MACRO'],
  randomSeed: 42
})
YIELD modelInfo
RETURN
  modelInfo.bestParameters AS bestParams,
  modelInfo.metrics.ACCURACY.test AS testAccuracy,
  modelInfo.metrics.F1_WEIGHTED.test AS testF1Weighted,
  modelInfo.metrics.F1_MACRO.test AS testF1Macro;

// Pipeline D-128
CALL gds.beta.pipeline.nodeClassification.create('pipeline-D-128');
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-128', { maxEpochs: 100, penalty: 0.001 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-128', { maxEpochs: 100, penalty: 0.01 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-128', { maxEpochs: 100, penalty: 0.1 });
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-D-128', { maxEpochs: 100, penalty: 1.0 });
CALL gds.beta.pipeline.nodeClassification.selectFeatures('pipeline-D-128', ['embedding128']);
CALL gds.beta.pipeline.nodeClassification.configureSplit('pipeline-D-128', { testFraction: 0.2, validationFolds: 5 });

CALL gds.beta.pipeline.nodeClassification.train('social-full', {
  pipeline: 'pipeline-D-128',
  targetNodeLabels: ['Person'],
  modelName: 'model-D-128',
  targetProperty: 'isInvolved',
  metrics: ['ACCURACY', 'F1_WEIGHTED', 'F1_MACRO'],
  randomSeed: 42
})
YIELD modelInfo
RETURN
  modelInfo.bestParameters AS bestParams,
  modelInfo.metrics.ACCURACY.test AS testAccuracy,
  modelInfo.metrics.F1_WEIGHTED.test AS testF1Weighted,
  modelInfo.metrics.F1_MACRO.test AS testF1Macro;

// =============================================================================
// SECTION 17: PIPELINE E — NODE2VEC EMBEDDINGS
// All three variants (default, global, local) collapsed to F1-Macro 0.4861
// Attributed to insufficient graph density for random walk convergence
// =============================================================================

// Pipeline E-default
CALL gds.beta.pipeline.nodeClassification.create('pipeline-E-default');
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-E-default', { maxEpochs: 100, penalty: 0.01 });
CALL gds.beta.pipeline.nodeClassification.selectFeatures('pipeline-E-default', ['embeddingN2V']);
CALL gds.beta.pipeline.nodeClassification.configureSplit('pipeline-E-default', { testFraction: 0.2, validationFolds: 5 });

CALL gds.beta.pipeline.nodeClassification.train('social-full', {
  pipeline: 'pipeline-E-default',
  targetNodeLabels: ['Person'],
  modelName: 'model-E-default',
  targetProperty: 'isInvolved',
  metrics: ['ACCURACY', 'F1_WEIGHTED', 'F1_MACRO'],
  randomSeed: 42
})
YIELD modelInfo
RETURN
  modelInfo.bestParameters AS bestParams,
  modelInfo.metrics.ACCURACY.test AS testAccuracy,
  modelInfo.metrics.F1_WEIGHTED.test AS testF1Weighted,
  modelInfo.metrics.F1_MACRO.test AS testF1Macro;

// Pipeline E-global (inOutFactor=0.5, explores outward)
CALL gds.beta.pipeline.nodeClassification.create('pipeline-E-global');
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-E-global', { maxEpochs: 100, penalty: 0.01 });
CALL gds.beta.pipeline.nodeClassification.selectFeatures('pipeline-E-global', ['embeddingN2V_global']);
CALL gds.beta.pipeline.nodeClassification.configureSplit('pipeline-E-global', { testFraction: 0.2, validationFolds: 5 });

CALL gds.beta.pipeline.nodeClassification.train('social-full', {
  pipeline: 'pipeline-E-global',
  targetNodeLabels: ['Person'],
  modelName: 'model-E-global',
  targetProperty: 'isInvolved',
  metrics: ['ACCURACY', 'F1_WEIGHTED', 'F1_MACRO'],
  randomSeed: 42
})
YIELD modelInfo
RETURN
  modelInfo.bestParameters AS bestParams,
  modelInfo.metrics.ACCURACY.test AS testAccuracy,
  modelInfo.metrics.F1_WEIGHTED.test AS testF1Weighted,
  modelInfo.metrics.F1_MACRO.test AS testF1Macro;

// Pipeline E-local (inOutFactor=2.0, stays local)
CALL gds.beta.pipeline.nodeClassification.create('pipeline-E-local');
CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipeline-E-local', { maxEpochs: 100, penalty: 0.01 });
CALL gds.beta.pipeline.nodeClassification.selectFeatures('pipeline-E-local', ['embeddingN2V_local']);
CALL gds.beta.pipeline.nodeClassification.configureSplit('pipeline-E-local', { testFraction: 0.2, validationFolds: 5 });

CALL gds.beta.pipeline.nodeClassification.train('social-full', {
  pipeline: 'pipeline-E-local',
  targetNodeLabels: ['Person'],
  modelName: 'model-E-local',
  targetProperty: 'isInvolved',
  metrics: ['ACCURACY', 'F1_WEIGHTED', 'F1_MACRO'],
  randomSeed: 42
})
YIELD modelInfo
RETURN
  modelInfo.bestParameters AS bestParams,
  modelInfo.metrics.ACCURACY.test AS testAccuracy,
  modelInfo.metrics.F1_WEIGHTED.test AS testF1Weighted,
  modelInfo.metrics.F1_MACRO.test AS testF1Macro;