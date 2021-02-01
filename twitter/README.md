# Synthetic Organizational Graph Dataset (Twitter)

This folder contains the graph datasets we synthesized using the Twitter egoNet from SNAP. These are directed graphs created from the original source dataset available at https://snap.stanford.edu/data/ego-Twitter.html. 
The files consist of an attribute file of one-hot encoded vectors as well as an edgelist file.

## Scenarios
Overall we have created 9 scenarios of 3 organizations having social graphs by varying the number of common vertices between them (And therefore the edges also) as well as the number of common attributes between them.

These varying degrees are labeled as LOW (~20%), MID (~50%), and HIGH (~80%) which are relative to the source dataset. 
Each combination of the varying common vertices and common attributes fall into a single scenario (Eg: LOW common vertices, MID common attributes).

Given the set of nodes (including the common nodes), the edgelist remains the same across the varying number of common attributes. 
For example, both `<MID common nodes, LOW common attributes>` scenario and `<MID common nodes, HIGH common attributes>` share the same edgelist.

## Files
### Edgelist
Edges follow the structure of `<user 1 id> <user 2 id>` where `user 1` follows `user 2` and their ids are space separated. 

`edgelistTwitter<#>.zip` contains the edgelist of all the nodes belonging to an organization.

`edgeListCommonEntityTwitter<#>.zip` contains the edgelist of the common nodes between the organizations.

### Attrinutes
Attribute files follow the structure of `<user id> <feature 1 value> ... <feature n value>` where features are space separated and are binary feature attributes extracted from the original ego dataset. 
All organizations combined have ~1500 attributes which are selected based on frequency of appearance and a predetermined threshold.

`org<#>.zip` contains the node attribute files and header file for that organization

`commonEntity.zip` contains the combined atrributes of all organizations for the common nodes

`global.zip` contains the combiedn attributes of all organizations for all the nodes (Note that this view is unobtainable in a real life scenario)
