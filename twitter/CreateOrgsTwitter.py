import networkx as nx
import glob
from collections import Counter
from nltk.corpus import stopwords
from random import random, seed
import csv
from sys import argv
import gc

featureList = []
featureCounter = 0
attributeDict = {}

logStr = ""

scn = ""
if "high" in argv[1].lower():
    scn = "HIGH"
elif "mid" in argv[1].lower():
    scn = "MID"
if "low" in argv[1].lower():
    scn = "LOW"


def log(*param, logMode=True):
    global logStr
    s = ""
    for p in param:
        s += str(p) + " "
    logStr += s + "\n"
    if logMode:
        print(s)


log("Converting to graph")
fh = open("twitter_combined.txt", 'rb')
graph = nx.read_edgelist(fh, create_using=nx.DiGraph())
fh.close()

# Calculate new vertex IDs
vertexIDMap = {}
vertexCounter = 0
for v in graph.nodes():
    if v not in vertexIDMap.keys():
        vertexIDMap[v] = vertexCounter
        vertexCounter = vertexCounter + 1

# Replace existing node labels with new labels
vertexIDMap = {str(k): str(val) for k, val in vertexIDMap.items()}
graph = nx.relabel_nodes(graph, {str(k): str(val) for k, val in vertexIDMap.items()})

# Load partitioned graphs
log("\n\n\n")
log("Reading partitions")
fh = open("3_0", 'rb')
g1 = nx.read_edgelist(fh, create_using=nx.DiGraph())
fh.close()
V1 = set(g1.nodes())

fh = open("3_1", 'rb')
g2 = nx.read_edgelist(fh, create_using=nx.DiGraph())
fh.close()
V2 = set(g2.nodes())

fh = open("3_2", 'rb')
g3 = nx.read_edgelist(fh, create_using=nx.DiGraph())
fh.close()
V3 = set(g3.nodes())

# Read common nodes of organisations
with open(argv[1], "r") as f:
    reader = csv.reader(f, delimiter="\n")
    commonNodes = list(reader)
commonNodes = set([v[0] for v in commonNodes])

log("\n\n\n")
log("Preparing org nodes\n")
# Org 1 nodes
organization_nodes1 = V1.union(commonNodes)

# Org 2 nodes
organization_nodes2 = V2.union(commonNodes)

# Org 3 nodes
organization_nodes3 = V3.union(commonNodes)

G = organization_nodes1.union(organization_nodes2).union(organization_nodes3)
log("global nodes:", len(G))
log("org 1 nodes:", len(organization_nodes1))
log("org 2 nodes:", len(organization_nodes2))
log("org 3 nodes:", len(organization_nodes3))

# Read features
log("\n\n\n")
log("Reading feature headers")
featNameFileList = [f for f in glob.glob("twitter/*.featnames")]
featuresInEachFile = {}
egoNet = {}
for f in featNameFileList:
    featureDictPerFile = {}
    lineCout = 0
    extracted = f.strip().split('.')
    index = vertexIDMap[extracted[0].split("/")[-1]]
    # print(index)
    egoNet[index] = []
    fh1 = open(str(f), 'r', encoding="utf8")
    for line in fh1:
        featureName = line.strip().split(' ', 1)[-1]
        term = featureName.split(':', 1)[-1]
        if term in set(stopwords.words("english")):
            continue
        else:
            featureList.append(featureName)
            egoNet[index].append(featureName)
            featureDictPerFile[lineCout] = featureName
        lineCout = lineCout + 1
    featuresInEachFile[index] = featureDictPerFile
    fh1.close()

log("Feature headers read")
# Compute feature counts for unique nodes of orgs
orgNet = {k: [] for k in [1, 2, 3]}
for v in V1:
    if v in egoNet:
        orgNet[1] += egoNet[str(v)]
for v in V2:
    if v in egoNet:
        orgNet[2] += egoNet[str(v)]
for v in V3:
    if v in egoNet:
        orgNet[3] += egoNet[str(v)]

orgFeatCounts = {k: Counter(orgNet[k]) for k in orgNet}

options = {
    "HIGH": 45,  # ~1200 common features out of 1500
    "MID": 59,  # ~750 common features out of 1500
    "LOW": 93,  # ~300 common features out of 1500
}

for opt, thresh in options.items():
    log("\n\n\n")
    log("Compute common features:", opt, "threshold:", thresh)
    featureCounts = Counter(featureList)
    c = 0
    selectedFeatures = {}
    for feat, cnt in featureCounts.items():
        if cnt > thresh:  # 93, 120, 185
            selectedFeatures[feat] = c
            # log(feat + " -- " + str(featureCounts[feat]))
            c = c + 1
    log("selected common attr:", c)

    for k in selectedFeatures:
        orgFeatCounts[1].pop(k, None)
        orgFeatCounts[2].pop(k, None)
        orgFeatCounts[3].pop(k, None)

    selectedFeatures1 = dict(selectedFeatures.items())
    c = len(selectedFeatures)
    orgMostCommon1 = orgFeatCounts[1].most_common((1500 - len(selectedFeatures)) // 3)
    for feat, count in orgMostCommon1:
        selectedFeatures1[feat] = c
        c += 1

    selectedFeatures2 = dict(selectedFeatures.items())
    c = len(selectedFeatures)
    orgMostCommon2 = orgFeatCounts[2].most_common((1500 - len(selectedFeatures)) // 3)
    for feat, count in orgMostCommon2:
        selectedFeatures2[feat] = c
        c += 1

    selectedFeatures3 = dict(selectedFeatures.items())
    c = len(selectedFeatures)
    orgMostCommon3 = orgFeatCounts[3].most_common((1500 - len(selectedFeatures)) // 3)
    for feat, count in orgMostCommon3:
        selectedFeatures3[feat] = c
        c += 1

    globalFeatures = dict(selectedFeatures.items())
    c = len(selectedFeatures)
    for feat, count in orgMostCommon1 + orgMostCommon2 + orgMostCommon3:
        globalFeatures[feat] = c
        c += 1

    log("global attr:", len(globalFeatures))
    log("org 1 attr:", len(selectedFeatures1))
    log("org 2 attr:", len(selectedFeatures2))
    log("org 3 attr:", len(selectedFeatures3))

    log("\n\n\nInstantiating node feature vectors", opt)
    attributeDict1 = {v: [0] * len(selectedFeatures1) for v in organization_nodes1}
    attributeDict2 = {v: [0] * len(selectedFeatures2) for v in organization_nodes2}
    attributeDict3 = {v: [0] * len(selectedFeatures3) for v in organization_nodes3}
    attributeDict = {v: [0] * len(globalFeatures) for v in G}

    featFileList = [f for f in glob.glob("twitter/*.feat")]

    for f in featFileList:
        fh1 = open(str(f), 'r', encoding="utf8")
        for line in fh1:
            extracted = line.strip().split(' ')
            nodeID = vertexIDMap[extracted[0]]
            index = vertexIDMap[f.strip().split('.')[0].split("/")[-1]]
            extractedFeatures = extracted[1:]
            correspondingFeatureDict = featuresInEachFile[index]
            for k, feat in correspondingFeatureDict.items():
                if feat in selectedFeatures1 and nodeID in organization_nodes1:
                    attributeDict1[nodeID][selectedFeatures1[feat]] = str(extractedFeatures[k])
                if feat in selectedFeatures2 and nodeID in organization_nodes2:
                    attributeDict2[nodeID][selectedFeatures2[feat]] = str(extractedFeatures[k])
                if feat in selectedFeatures3 and nodeID in organization_nodes3:
                    attributeDict3[nodeID][selectedFeatures3[feat]] = str(extractedFeatures[k])
                if feat in globalFeatures and nodeID in organization_nodes1:
                    attributeDict[nodeID][globalFeatures[feat]] = str(extractedFeatures[k])
        fh1.close()

    log(f"Writing org 1 selected features to file", opt)
    with open(f"{scn}CommonEntity/{opt}CommonAttr/selectedAttributes1.txt", "w") as selectedFeatureFile1:
        writer = csv.writer(selectedFeatureFile1, delimiter=" ")
        writer.writerows([[feature.replace('"', '')] for feature in selectedFeatures1.keys()])

    log(f"Writing org 2 selected features to file", opt)
    with open(f"{scn}CommonEntity/{opt}CommonAttr/selectedAttributes2.txt", "w") as selectedFeatureFile2:
        writer = csv.writer(selectedFeatureFile2, delimiter=" ")
        writer.writerows([[feature.replace('"', '')] for feature in selectedFeatures2.keys()])

    log(f"Writing org 3 selected features to file", opt)
    with open(f"{scn}CommonEntity/{opt}CommonAttr/selectedAttributes3.txt", "w") as selectedFeatureFile3:
        writer = csv.writer(selectedFeatureFile3, delimiter=" ")
        writer.writerows([[feature.replace('"', '')] for feature in selectedFeatures3.keys()])

    log(f"Writing global selected features to file", opt)
    with open(f"{scn}CommonEntity/{opt}CommonAttr/selectedAttributes.txt", "w") as selectedFeatureFile:
        writer = csv.writer(selectedFeatureFile, delimiter=" ")
        writer.writerows([[feature.replace('"', '')] for feature in globalFeatures.keys()])

    # Write organizational attributes to file
    log("\n\n")
    log(f"Writing organization 1 attributes", opt)
    with open(f"{scn}CommonEntity/{opt}CommonAttr/attributesTwitter1.txt", 'w') as attFile1:
        for k, v in attributeDict1.items():
            attFile1.write(str(k))
            for i in v:
                attFile1.write("\t" + str(i))
            attFile1.write("\n")

    log(f"Writing organization 2 attributes", opt)
    with open(f"{scn}CommonEntity/{opt}CommonAttr/attributesTwitter2.txt", 'w') as attFile2:
        for k, v in attributeDict2.items():
            attFile2.write(str(k))
            for i in v:
                attFile2.write("\t" + str(i))
            attFile2.write("\n")

    log(f"Writing organization 3 attributes", opt)
    with open(f"{scn}CommonEntity/{opt}CommonAttr/attributesTwitter3.txt", 'w') as attFile3:
        for k, v in attributeDict3.items():
            attFile3.write(str(k))
            for i in v:
                attFile3.write("\t" + str(i))
            attFile3.write("\n")

    log(f"Writing Global attributes", opt)
    with open(f"{scn}CommonEntity/{opt}CommonAttr/globalAttrTwitter.txt", 'w') as attFile3:
        for k, v in attributeDict.items():
            attFile3.write(str(k))
            for i in v:
                attFile3.write("\t" + str(i))
            attFile3.write("\n")

    log(f"Writing Common entity attributes", opt)
    with open(f"{scn}CommonEntity/{opt}CommonAttr/commonEntityAttrTwitter.txt", 'w') as attFile3:
        for k in commonNodes:
            v = attributeDict[k]
            attFile3.write(str(k))
            for i in v:
                attFile3.write("\t" + str(i))
            attFile3.write("\n")

    del attributeDict, attributeDict1, attributeDict2, attributeDict3
    gc.collect()

# Create edgelists
log("\n\n\n")
log("Creating organisation edgelists")

# Keep edges of nodes belonging to an original partition
# Randomly remove edges of nodes of edges of the common node subgraph outside of relevant organization partition
graph_org1 = graph.subgraph(organization_nodes1).copy()
graph_org2 = graph.subgraph(organization_nodes2).copy()
graph_org3 = graph.subgraph(organization_nodes3).copy()

partitionEdges1 = graph_org1.subgraph(V1).edges()
partitionEdges2 = graph_org2.subgraph(V2).edges()
partitionEdges3 = graph_org3.subgraph(V3).edges()

log("Partition edges 1", len(partitionEdges1))
log("Partition edges 2", len(partitionEdges2))
log("Partition edges 3", len(partitionEdges3))

log("\n\n\n")

seed(7)
edgesToRemove1 = set(filter(lambda x: x not in partitionEdges1 and random() > 0.5, graph_org1.edges()))
edgesToRemove2 = set(filter(lambda x: x not in partitionEdges2 and random() > 0.5, graph_org2.edges()))
edgesToRemove3 = set(filter(lambda x: x not in partitionEdges3 and random() > 0.5, graph_org3.edges()))

graph_org1.remove_edges_from(edgesToRemove1)
graph_org2.remove_edges_from(edgesToRemove2)
graph_org3.remove_edges_from(edgesToRemove3)

log("Preparing org 1 edgelist")
organization_edgelist1 = set(graph_org1.edges())
log("Preparing org 2 edgelist")
organization_edgelist2 = set(graph_org2.edges())
log("Preparing org 3 edgelist")
organization_edgelist3 = set(graph_org3.edges())
log("Global graph edgelist")
global_graph = organization_edgelist1.union(organization_edgelist2).union(organization_edgelist3)

log("global edges:", len(global_graph))
log("org 1 edges:", len(organization_edgelist1))
log("org 2 edges:", len(organization_edgelist2))
log("org 3 edges:", len(organization_edgelist3))

log("Writing global graph edgelist")
edgeFile = open(f"{scn}CommonEntity/edgeListTwitter.txt", 'w')
for edge in global_graph:
    edgeFile.write(str(edge[0]) + "\t" + str(edge[1]))
    edgeFile.write("\n")
edgeFile.close()

log("Writing organization 1 edgelist")
with open(f"{scn}CommonEntity/edgeListTwitter1.txt", 'w') as edgeFile:
    for edge in organization_edgelist1:
        edgeFile.write(str(edge[0]) + "\t" + str(edge[1]))
        edgeFile.write("\n")
    edgeFile.close()

log("Writing organization 2 edgelist")
with open(f"{scn}CommonEntity/edgeListTwitter2.txt", 'w') as edgeFile:
    for edge in organization_edgelist2:
        edgeFile.write(str(edge[0]) + "\t" + str(edge[1]))
        edgeFile.write("\n")
    edgeFile.close()

log("Writing organization 3 edgelist")
with open(f"{scn}CommonEntity/edgeListTwitter3.txt", 'w') as edgeFile:
    for edge in organization_edgelist3:
        edgeFile.write(str(edge[0]) + "\t" + str(edge[1]))
        edgeFile.write("\n")
    edgeFile.close()

log("Writing organization 1 common nodes' edgelist")
with open(f"{scn}CommonEntity/edgeListCommonEntityTwitter1.txt", 'w') as edgeFile:
    for edge in graph_org1.subgraph(commonNodes).edges():
        edgeFile.write(str(edge[0]) + "\t" + str(edge[1]))
        edgeFile.write("\n")
    edgeFile.close()

log("Writing organization 2 common nodes' edgelist")
with open(f"{scn}CommonEntity/edgeListCommonEntityTwitter2.txt", 'w') as edgeFile:
    for edge in graph_org2.subgraph(commonNodes).edges():
        edgeFile.write(str(edge[0]) + "\t" + str(edge[1]))
        edgeFile.write("\n")
    edgeFile.close()

log("Writing organization 3 common nodes' edgelist")
with open(f"{scn}CommonEntity/edgeListCommonEntityTwitter3.txt", 'w') as edgeFile:
    for edge in graph_org3.subgraph(commonNodes).edges():
        edgeFile.write(str(edge[0]) + "\t" + str(edge[1]))
        edgeFile.write("\n")
    edgeFile.close()

with open(f"{scn}CommonEntity/log.txt", 'w') as logfile:
    logfile.write(logStr)
