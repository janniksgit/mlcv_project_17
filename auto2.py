"""
ISBI 2012 Simple 2D Multicut Pipeline
======================================

Here we segment neuro data as in  :cite:`beier_17_multicut`.
In fact, this is a simplified version of :cite:`beier_17_multicut`.
We start from an distance transform watershed
over-segmentation.
We compute a RAG and features for all edges.
Next, we learn the edge probabilities
with a random forest classifier.
The predicted edge probabilities are
fed into multicut objective.
This is optimized with an ILP solver (if available).
This results into a ok-ish learned segmentation
for the ISBI 2012 dataset.

This example will download a about 400 MB large zip file
with the dataset and precomputed results from :cite:`beier_17_multicut`


"""


# multi purpose
import numpy

# plotting
import pylab

# to download data and unzip it
import os
import urllib.request
import zipfile

# to read the tiff files
import skimage.io
import skimage.filters
import skimage.morphology

# classifier
from sklearn.ensemble import RandomForestClassifier

# needed parts of nifty
import sys
sys.path.append('/Users/jannikkossen/Dropbox-Uni/Dropbox/Studium/Semester8/MachineLearningForComputerVision/Project/nifty/build/python')

import nifty
import nifty.segmentation
import nifty.filters
import nifty.graph
import nifty.graph.rag
import nifty.ground_truth
import nifty.graph.optimization.multicut

#############################################################
# Download  ISBI 2012:
# =====================
# Download the  ISBI 2012 dataset 
# and precomputed results form :cite:`beier_17_multicut`
# and extract it in-place.
fname = "data.zip"
url = "http://files.ilastik.org/multicut/NaturePaperDataUpl.zip"
if not os.path.isfile(fname):
    print("Downloading files...")
    urllib.request.urlretrieve(url, fname)
    zip = zipfile.ZipFile(fname)
    zip.extractall()


#############################################################
# Setup Datasets:
# =================
# load ISBI 2012 raw and probabilities
# for train and test set
# and the ground-truth for the train set

# debugging: control how many plots we get
plot_multiplier = 4

# debugging: only use part of the training images
lim = 4

# we use part of the training images as ground truth    
split = 27

rawDsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/raw_train.tif')[0:lim],
    'test' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/raw_train.tif')[split:],
}

# read pmaps and convert to 01 pmaps
pmapDsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/probabilities_train.tif')[0:lim],
    'test' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/probabilities_train.tif')[split:],
}
pmapDsets = {
    'train' : pmapDsets['train'].astype('float32')/255.0,
    'test' : pmapDsets['test'].astype('float32')/255.0
}
gtDsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/groundtruth.tif')[0:lim],
    'test'  : skimage.io.imread('NaturePaperDataUpl/ISBI2012/groundtruth.tif')[split:]
}

computedData = {
    'train' : [{} for z in range(rawDsets['train'].shape[0])],
    'test'  : [{} for z in range(rawDsets['test'].shape[0])]
}

######################################d
#  Compute Normal Image Features:
#  ===================================

def computeFeatures(raw, pmap, rag):
    """
        Computes features for one image.
    """
    uv = rag.uvIds()
    nrag = nifty.graph.rag

    # list of all edge features we fill 
    feats = []

    # helper function to convert 
    # node features to edge features
    def nodeToEdgeFeat(nodeFeatures):
        uF = nodeFeatures[uv[:,0], :]
        vF = nodeFeatures[uv[:,1], :]
        feats = [ numpy.abs(uF-vF), uF + vF, uF *  vF,
                 numpy.minimum(uF,vF), numpy.maximum(uF,vF)]
        return numpy.concatenate(feats, axis=1)

    # accumulate features from raw data
    fRawEdge, fRawNode = nrag.accumulateStandartFeatures(rag=rag, data=raw,
        minVal=0.0, maxVal=255.0, numberOfThreads=1)
    feats.append(fRawEdge)
    feats.append(nodeToEdgeFeat(fRawNode))

    # accumulate data from pmap
    fPmapEdge, fPmapNode = nrag.accumulateStandartFeatures(rag=rag, data=pmap, 
        minVal=0.0, maxVal=1.0, numberOfThreads=1)
    feats.append(fPmapEdge)
    feats.append(nodeToEdgeFeat(fPmapNode))

    # accumulate node and edge features from
    # superpixels geometry 
    fGeoEdge = nrag.accumulateGeometricEdgeFeatures(rag=rag, numberOfThreads=1)
    feats.append(fGeoEdge)

    fGeoNode = nrag.accumulateGeometricNodeFeatures(rag=rag, numberOfThreads=1)
    feats.append(nodeToEdgeFeat(fGeoNode))

    return numpy.concatenate(feats, axis=1)


#############################################################
#  Over-segmentation, RAG & Extract Features:
#  ============================================
# 
#  Compute:
#   *    Over-segmentation  with distance transform watersheds.
#   *    Construct a region adjacency graph (RAG)
#   *    Extract features for all edges in the graph
#   *    Map the ground truth to the edges in the graph.
#        (only for the training set)
#
 
for ds in ['train', 'test']:
    
    rawDset = rawDsets[ds]
    pmapDset = pmapDsets[ds]
    gtDset = gtDsets[ds]
    dataDset = computedData[ds]

    # for each slice
    for z in range(rawDset.shape[0]):   
             
        data = dataDset[z]

        # get raw and pmap slice
        raw  = rawDset[z, ... ]
        pmap = pmapDset[z, ... ]

        # oversementation
        overseg = nifty.segmentation.distanceTransformWatersheds(pmap, threshold=0.3)
#        overseg = nifty.segmentation.distanceTransformWatersheds(pmap, threshold=0.99)
        overseg -= 1
        data['overseg'] = overseg

        # region adjacency graph
        rag = nifty.graph.rag.gridRag(overseg)
        data['rag'] = rag

        # compute features
        features = computeFeatures(raw=raw, pmap=pmap, rag=rag)

        data['features'] = features

        # map the gt to edge

        # the gt is on membrane level
        # 0 at membranes pixels
        # 1 at non-membrane pixels
        gtImage = gtDset[z, ...] 

        # local maxima seeds
        seeds = nifty.segmentation.localMaximaSeeds(gtImage)
        data['partialGt'] = seeds
        # growing map
        growMap = nifty.filters.gaussianSmoothing(1.0-gtImage, 1.0)
        growMap += 0.1*nifty.filters.gaussianSmoothing(1.0-gtImage, 6.0)
        gt = nifty.segmentation.seededWatersheds(growMap, seeds=seeds)


        # map the gt to the edges
        overlap = nifty.ground_truth.overlap(segmentation=overseg, 
                                   groundTruth=gt)

        # edge gt
        edgeGt = overlap.differentOverlaps(rag.uvIds())
        data['gt'] = gt
        data['edgeGt'] = edgeGt


        # plot each "plot_multiplier"th 
        if z  % plot_multiplier == 0 :
            figure = pylab.figure()
            figure.suptitle('Training Set Slice %d'%z, fontsize=20)
            #fig = matplotlib.pyplot.gcf()
            figure.set_size_inches(19.5, 10.5)
            figure.add_subplot(3, 2, 1)
            pylab.imshow(raw, cmap='gray')
            pylab.title("Raw data %s"%(ds))
            figure.add_subplot(3, 2, 2)
            pylab.imshow(pmap, cmap='gray')
            pylab.title("Membrane pmap %s"%(ds))
            figure.add_subplot(3, 2, 3)
            pylab.imshow(nifty.segmentation.segmentOverlay(raw, overseg, 0.2, thin=False))
            pylab.title("Superpixels %s"%(ds))
            figure.add_subplot(3, 2, 4)
            pylab.imshow(seeds, cmap=nifty.segmentation.randomColormap(zeroToZero=True))
            pylab.title("Partial ground truth %s" %(ds))
            figure.add_subplot(3, 2, 5)
            pylab.imshow(nifty.segmentation.segmentOverlay(raw, gt, 0.2, thin=False))
            pylab.title("Dense ground truth %s" %(ds))            
            figure.add_subplot(3,2,6)
            pylab.imshow(gt, cmap=nifty.segmentation.randomColormap(zeroToZero=True))
            pylab.title("Dense ground truth %s" %(ds))
            pylab.tight_layout()
            pylab.savefig('figs/building_%sImg%d.pdf'%(ds, z))


##############################
#  More Helper Functions:
#  ===========================
        
def buildTrainingSet(lims, rf = None, iteration = 1):   
    """
    Build training set from computed features.
    Filter out bad ground truth edges.
    Input:
        lims: array [a,b], subset of the training set used for training
        rf : random forest, only if we're not in the first iteration
    Output:
        trainingSet: dictionary
    """         
    ds = 'train'
    dataDset = computedData[ds]
    rawDset = rawDsets[ds]
    trainingSet = {'features':[],'labels':[]}
    
    # for each slice
    # trainsplit
    #for z in range(rawDset.shape[0]): 
    print("Built features from train images in range [%d,%d]" %(lims[0], lims[1]))
    for z in range(*lims):   
        
        data = dataDset[z]
        edgeGt = data['edgeGt']    
        features = data['features']
        
        if rf is not None:
            rag = data['rag']
            raw = rawDset[z,...]
            # get all edge probabilities
            predictions = rf.predict_proba(features)[:,1]
                        
            # solve multicut b/c we want to include multicut result in 
            # prediction (again, we need to solve multicut for predictions from
            # first random forest for second train images)
            MulticutObjective = rag.MulticutObjective
            eps =  0.00001
            p1 = numpy.clip(predictions, eps, 1.0 - eps) 
            weights = numpy.log((1.0-p1)/p1)
            objective = MulticutObjective(rag, weights)
            solver = MulticutObjective.greedyAdditiveFactory().create(objective)
            mc_results = solver.optimize(visitor=MulticutObjective.verboseVisitor())
            
            # build new features from random forest prediction
            new_feats = feat_from_edge_prob(rag, raw, predictions, mc_results)
            
            # features = new_feats
            # option: add to old features
            # note: change also in predictAndCut
            features = numpy.hstack([new_feats, features])
            
        # we use only edges which have
        # a high certainty
        # (Only use the good edges of our data! Has nothing to with our 
        # features.)
        where1 = numpy.where(edgeGt > 0.85)[0]
        where0 = numpy.where(edgeGt < 0.15)[0]

        trainingSet['features'].append(features[where0,:])
        trainingSet['features'].append(features[where1,:])
        trainingSet['labels'].append(numpy.zeros(len(where0)))
        trainingSet['labels'].append(numpy.ones(len(where1)))
    
    return trainingSet

def predictAndCut(rf, iteration, ds = 'test', plotting = 'on'):
    """
        Solves multicut problem from trained random forest.
    Args:
        rf: the random forest object
        ds: the data set specification, i.e. 'test' or 'train'
        iteration: integer used for saving the multicut result (might use it
                    as a feature later on)
        plotting: 'on' or 'off'

    Output:
        segmentation images
    """
    
    rawDset = rawDsets[ds]
    pmapDset = pmapDsets[ds]
    dataDset = computedData[ds]

    # for each slice
    multicutResults = []
    print("Predict and Cut edges using %s images."%(ds))
    for z in range(rawDset.shape[0]):   
        data = dataDset[z]

        pmap = pmapDset[z,...]
        overseg = data['overseg']
        gt = data['gt']
        features = data['features']
        rag = data['rag']
        raw = rawDset[z,...]

        # need to also calculate new features
        # for test image
        # and just in case I ever forget this agin
        # these are NEW UNKNOWN images
        # dont reuse anything from training
        
        # if we have already applied rf, build additional features
        if iteration >= 2:
            # get all edge probabilities                        
            predictions = rf1.predict_proba(features)[:,1]
                        
            # solve multicut b/c we want to include multicut result in 
            # prediction (again, we need to solve multicut for predictions from
            # first random forest for second train images)
            MulticutObjective = rag.MulticutObjective
            eps =  0.00001
            p1 = numpy.clip(predictions, eps, 1.0 - eps) 
            weights = numpy.log((1.0-p1)/p1)
            objective = MulticutObjective(rag, weights)
            solver = MulticutObjective.greedyAdditiveFactory().create(objective)
            mc_results = solver.optimize(visitor=MulticutObjective.verboseVisitor())
            
            # build new features from random forest prediction
            new_feats = feat_from_edge_prob(rag, raw, predictions, mc_results)

            
            # features = new_feats
            # option: add to old features
            # note: change also in buildTraining
            features = numpy.hstack([new_feats, features])
            

        print("Predict weights from Random Forest with %d features per edge" 
              %(features.shape[1]))
        predictions = rf.predict_proba(features)[:,1]

        # setup multicut objective
        MulticutObjective = rag.MulticutObjective

        eps =  0.00001
        p1 = numpy.clip(predictions, eps, 1.0 - eps) 
        weights = numpy.log((1.0-p1)/p1)
        
        objective = MulticutObjective(rag, weights)

        # do multicut obtimization 
        solver = MulticutObjective.greedyAdditiveFactory().create(objective)

        arg = solver.optimize(visitor=MulticutObjective.verboseVisitor())
        multicutResults.append(arg)
        # save multicut result
        # data['arg%d'%iteration] = arg
        result = nifty.graph.rag.projectScalarNodeDataToPixels(rag, arg)
        
        # plot for each "multiplier"th slice
        if plotting == 'on' and z % plot_multiplier == 0:
            figure = pylab.figure() 
            figure.suptitle('Test Set Results Slice %d'%z, fontsize=20)
            #fig = matplotlib.pyplot.gcf()
            figure.set_size_inches(18.5, 10.5)
            figure.add_subplot(3, 2, 1)
            pylab.imshow(raw, cmap='gray')
            pylab.title("Raw data %s"%(ds))
            figure.add_subplot(3, 2, 2)
            pylab.imshow(pmap, cmap='gray')
            pylab.title("Membrane pmap %s"%(ds))
            figure.add_subplot(3, 2, 3)
            pylab.imshow(nifty.segmentation.segmentOverlay(raw, overseg, 0.2, thin=False))
            pylab.title("Superpixels %s"%(ds))
            figure.add_subplot(3, 2, 4)
            pylab.imshow(result, cmap=nifty.segmentation.randomColormap())
            pylab.title("Result seg. %s" %(ds))
            figure.add_subplot(3, 2, 5)
            pylab.imshow(nifty.segmentation.segmentOverlay(raw, result, 0.2, thin=False))
            pylab.title("Result seg. %s" %(ds))     
            figure.add_subplot(3,2,6)
            pylab.imshow(gt, cmap=nifty.segmentation.randomColormap(zeroToZero=True))
            pylab.title("Dense ground truth %s" %(ds))
            pylab.tight_layout()
            pylab.savefig('figs/testing_testImg%d_RF%d.pdf' %(z, iteration))
        
    return multicutResults

from numpy import random

def feat_from_edge_prob(rag, raw, edge_probs, mc_results):
    """
    
    Create new features from edge probabilities. Works on a single image. 
    Therefore pass rag and raw from one image!
    
    Args:
        rag: the graph (superpixel region adjacency graph)
        raw: raw image (shape maybe 1000x1000)
        edge_probs: shape: |rag.numberOfEdges|

    Output:
        new_feat: shape |rag.numberEdges, NUMBER_OF_NEW_FEAT|
    """

    edgeIndicators = edge_probs
    new_feats = []

    # trivial feature
    new_feats.append(edge_probs[:,None])

    #TODO: add multicut feature
    # new_feats.append(mc_results[:,None])
    
    import nifty.graph.agglo
    nagglo = nifty.graph.agglo

    edgeSizes = numpy.ones(shape=[rag.numberOfEdges])
    nodeSizes = numpy.ones(shape=[rag.numberOfNodes])
    
    # ucm features
    for r in (0.01,0.1,0.2,0.4,0.5, 0.8):

        clusterPolicy = nagglo.edgeWeightedClusterPolicyWithUcm(
            graph=rag, edgeIndicators=edgeIndicators,
            edgeSizes=edgeSizes, nodeSizes=nodeSizes, sizeRegularizer=r)

        agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy) 

        a_new_feat = agglomerativeClustering.runAndGetDendrogramHeight(verbose=False)

        # Test if we're using the right set of features
        #random_feat = random.rand(*a_new_feat.shape)
        #new_feats.append(random_feat[:,None])        
        new_feats.append(a_new_feat[:,None])

    new_feats = numpy.concatenate(new_feats, axis=1)
    return new_feats


###########################
# ----- MAIN CODE ---------
#  Auto Context Classifier
# =========================


## round 1

# build the first training set

# prevent overfitting: split up training set for the different random forests
train_split = 2

trainingSet1 = buildTrainingSet([0,train_split])
train_features1 = numpy.concatenate(trainingSet1['features'], axis=0)
train_labels1 = numpy.concatenate(trainingSet1['labels'], axis=0)

# train random forest 1
print("Train Random Forest1 with %d features per edge" %(trainingSet1['features'][0].shape[1]))
rf1 = RandomForestClassifier(n_estimators=200, oob_score=True)
rf1.fit(train_features1, train_labels1)
print("OOB SCORE 1", rf1.oob_score_)

# predict edges from random forest and solve multicut problem
multicutResults1 = predictAndCut(rf1, iteration = 1)
            
## round 2
# build the second training set
# now, features from the rf1 predictions will be included
trainingSet2 = buildTrainingSet([train_split, lim], rf1)
# debug: dont split up training set
# trainingSet2 = buildTrainingSet([0, train_split], rf1)
train_features2 = numpy.concatenate(trainingSet2['features'], axis=0)
train_labels2 = numpy.concatenate(trainingSet2['labels'], axis=0)

# retrain random forest 2 with these new features
print("Train Random Forest2 with %d features per edge" %(trainingSet2['features'][0].shape[1]))
rf2 = RandomForestClassifier(n_estimators=200, oob_score=True)
rf2.fit(train_features2, train_labels2)
print("OOB SCORE 2",rf2.oob_score_)

# predict edges from random forest and solve multicut problem
multicutResults2 = predictAndCut(rf2, iteration = 2)

# quick testing.
# grab first segmentation results from both rf and compare their content


#####
from nifty import ground_truth

def benchmark(multicutResults):
    """
    Input:
        list of segmentation images
    Output:
        mean precision and recall values over all images
    """
    
    ds = 'test'
    rawDset = rawDsets[ds]
    dataDset = computedData[ds]
    error = numpy.zeros([rawDset.shape[0], 2])
    # for each slice
    for z in range(rawDset.shape[0]):
            
        data = data = dataDset[z]
        partialGt = data['partialGt']
        rag = data['rag']
        multicutResult = multicutResults[z]
        seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, multicutResult)
        
        randError = ground_truth.RandError(partialGt, seg, ignoreDefaultLabel = True)
        variationError = ground_truth.VariationOfInformation(partialGt, seg, ignoreDefaultLabel = True)
        error[z,:] = [randError.error, variationError.value]
        
    
    mean_error = numpy.mean(error, axis = 0)
    

    return mean_error

mean_error1= benchmark(multicutResults1)
print(mean_error1)
mean_error2 = benchmark(multicutResults2)
print(mean_error2)

#    figure = pylab.figure() 
#    figure.add_subplot(2, 1, 1)
#    pylab.title('Ground Truth')
#    pylab.imshow(partialGt)
#    
#    figure.add_subplot(2, 1, 2)
#    pylab.title('Segmentation')
#    pylab.imshow(seg)
#    pylab.tight_layout()
#    pylab.savefig('figs/benchmarking_illustration.pdf')

# visualize(data['rag'], data['overseg'], data['edgeGt'], numpy.zeros(data['overseg'].shape) )


