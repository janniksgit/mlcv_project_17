import nifty.ground_truth

gt 
pred 
a = nifty.ground_truth.RandError(gt, pred)
a.randError


def feat_from_edge_prob(rag, raw, edge_probs):
    """[summary]
    
    [description]
    
    Args:
        rag: the graph (superpixel region adjacency graph)
        raw: raw image (shape maybe 1000x1000)
        edge_probs: shape: |rag.numberOfEdges|

    Output:
        new_feat: shape |rag.numberEdges, NUMBER_OF_NEW_FEAT|
    """
    
    new_feats = []

    # trivial feature
    new_feats.append(edge_probs[:,None])

    # ucm features

    for r in (0.01,0.1,0.2,0.4,0.5, 0.8):

        clusterPolicy = ngraph.agglo.edgeWeightedClusterPolicyWithUcm(
            graph=graph, edgeIndicators=edgeIndicators,
            edgeSizes=edgeSizes, nodeSizes=nodeSizes,sizeRegularizer=r)

        agglomerativeClustering = ngraph.agglo.agglomerativeClustering(clusterPolicy) 

        a_new_feat = agglomerativeClustering.runAndGetDendrogramHeight(verbose=False)
        
        new_feats.append(a_new_feat[:,None])

    # the multicut feature

    new_feats = numpy.concatenate(new_feats, axis=1)

