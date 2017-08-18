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


from visu import visualize
import numpy
import vigra
from scipy.ndimage.filters import maximum_filter
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
plotting = 'off'
# debugging: only use part of the training images

# we use part of the training images as ground truth    
split = 27
lim = 18

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

    # helper function to convert 
    # node features to edge features
def nodeToEdgeFeat(nodeFeatures, rag):
    uv = rag.uvIds()
    uF = nodeFeatures[uv[:,0], :]
    vF = nodeFeatures[uv[:,1], :]
    feats = [ numpy.abs(uF-vF), uF + vF, uF *  vF,
             numpy.minimum(uF,vF), numpy.maximum(uF,vF)]
    return numpy.concatenate(feats, axis=1)


def computeFeatures(raw, pmap, rag):
    """
        Computes features for one image.
    """
    
    nrag = nifty.graph.rag

    # list of all edge features we fill 
    feats = []

    # accumulate features from raw data
    fRawEdge, fRawNode = nrag.accumulateStandartFeatures(rag=rag, data=raw,
        minVal=0.0, maxVal=255.0, numberOfThreads=1)
    feats.append(fRawEdge)
    feats.append(nodeToEdgeFeat(fRawNode, rag))

    # accumulate data from pmap
    fPmapEdge, fPmapNode = nrag.accumulateStandartFeatures(rag=rag, data=pmap, 
        minVal=0.0, maxVal=1.0, numberOfThreads=1)
    feats.append(fPmapEdge)
    feats.append(nodeToEdgeFeat(fPmapNode, rag))

    # accumulate node and edge features from
    # superpixels geometry 
    fGeoEdge = nrag.accumulateGeometricEdgeFeatures(rag=rag, numberOfThreads=1)
    feats.append(fGeoEdge)

    fGeoNode = nrag.accumulateGeometricNodeFeatures(rag=rag, numberOfThreads=1)
    feats.append(nodeToEdgeFeat(fGeoNode, rag))

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
 
print("Compute needed data features (overseg, rag, edgeGt,..) ...")
      
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
#        overseg = nifty.segmentation.distanceTransformWatersheds(pmap, threshold=0.05)

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
        if z  % plot_multiplier == 0 and plotting == 'on' :
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



print("Stuff computed..")

##############################
#  More Helper Functions:
#  ===========================


def buildFeatures(subsection_lims, rfs = []):   
    """
    Build features for training on training set.
    If random forest is specified (so after first iteration), features from 
    that random forest will also be calculated.
    Filter out bad ground truth edges.
    Input:
        subsection_lims: array [a,b], subset of the training set used for training
        
    Output:
        training_features, training_labels -> plug directly into RF
    """         
    ds = 'train'
    dataDset = computedData[ds]
    rawDset = rawDsets[ds]
    trainingSet = {'features':[],'labels':[]}
    
    # for each slice
    # trainsplit
    #for z in range(rawDset.shape[0]): 
    print("-- Built features from %s images in range [%d,%d] for rf%d ..." 
          %(ds, subsection_lims[0], subsection_lims[1], len(rfs)))
    for z in range(*subsection_lims):   
        data = dataDset[z]
        edgeGt = data['edgeGt']    
        img_feats = data['features']
        
        # If we are not in the first round, we want to create extra features
        # from the predictinos of the precious random forest.
        
        # Structure of loop. 
        # Start by building features for RF1:
        # use RF0 to predict edge probs, from those, build new features for RF1
        # update features value
        # Now: build features fro RF2:
        # use RF1 to predict edge_probs with features for RF1 from RF0, from 
        # those, build new features for RF2, update features value
        # feats for RF3: predict with RF2 with features generated
                
        if len(rfs) is not 0:
            # initialise with standard features
            features = img_feats
            for i, rf in enumerate(rfs):
                print("---- Predicting with rf%d on image %d" %(i, z))
                rag = data['rag']
                raw = rawDset[z,...]
                # get all edge probabilities                        
                # solve multicut b/c we want to include multicut result in 
                # prediction (again, we need to solve multicut for predictions from
                # first random forest for second train images)
                edge_probs = rf.predict_proba(features)[:,1]
                #edge_probs = rf.predict(features)

                MulticutObjective = rag.MulticutObjective
                eps =  0.00001
                p1 = numpy.clip(edge_probs, eps, 1.0 - eps) 
                weights = numpy.log((1.0-p1)/p1)
                objective = MulticutObjective(rag, weights)
                solver = MulticutObjective.greedyAdditiveFactory().create(objective)
                mc_results = solver.optimize(visitor=MulticutObjective.verboseVisitor())
                
                # build new features from random forest prediction
                rf_feats = feat_from_edge_prob(rag, raw, edge_probs, mc_results)
                
                # update features for next random forest
                #features = rf_feats
                features = numpy.hstack([rf_feats, img_feats])    
                # option: add to old features

        else:
            features = img_feats
            
            
            
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

    train_features = numpy.concatenate(trainingSet['features'], axis=0)
    train_labels = numpy.concatenate(trainingSet['labels'], axis=0)
    print("-- Finished building features of shape", train_features.shape, "..." )
    return train_features, train_labels


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
        new_feats.append(a_new_feat[:,None])

#        Test if we're using the right set of features
#        from numpy import random
#        print("WATCH OUT; RANDOM FEATURES!")
#        random_feat = random.rand(*a_new_feat.shape)
#        new_feats.append(random_feat[:,None])        
        
#    numpy.save('edgepred',edge_probs)
    vispred = visualize(rag, overseg, edge_probs, numpy.zeros(overseg.shape))
#    numpy.save('vispred',vispred)
    
    #vispred = maximum_filter(vispred)
     
    #apply several filters on visualization of the prediction (a max filter is applied because Thorsten said so)
     
    imgs = []
    for sigma in [2.0, 4.0, 6.0]:
        res = vigra.filters.gaussianSmoothing(vispred, sigma)   
#        numpy.save('resg',res)
        imgs.append(res)
         
#    for sigma in [2.,4.,6.]:
#        res = vigra.filters.hessianOfGaussianEigenvalues(vispred, sigma)
#        numpy.save('resh',res)
#        imgs.append(res[...,0])
#    
#    for inscale in [1.0,2.0,3.0]:
#        res = vigra.filters.structureTensorEigenvalues(vispred,inscale,inscale*5.)
#        imgs.append(res[...,0])
    #imgs = numpy.concatenate(imgs,axis =1)    

    nrag = nifty.graph.rag
    uv = rag.uvIds()
    def nodeToEdgeFeat(nodeFeatures):
        uF = nodeFeatures[uv[:,0], :]
        vF = nodeFeatures[uv[:,1], :]
        feats = [ numpy.abs(uF-vF), uF + vF, uF *  vF,
                 numpy.minimum(uF,vF), numpy.maximum(uF,vF)]
        return numpy.concatenate(feats, axis=1)
    
#    numpy.save('imgs',imgs)
    for img in imgs:
        fRawEdge, fRawNode = nrag.accumulateStandartFeatures(rag=rag, data=img,
        minVal=0.0, maxVal=1.0, numberOfThreads=1)
        new_feats.append(fRawEdge)
        new_feats.append(nodeToEdgeFeat(fRawNode))
    
    new_feats = numpy.concatenate(new_feats, axis=1)
#    numpy.save('new_feats',new_feats)    
    return new_feats


###########################
# ----- MAIN CODE ---------
#  Auto Context Classifier
# =========================f


## LOOPY Forest Training
    
# prevent overfitting: split up training set for the different random forests
rfs = []
num_rfs = 3

# Random Sample
#from numpy import random
#num_per_set = 4
#tmps = random.choice(range(0, lim-num_per_set), num_rfs, replace = False)

# Make it myself, tmps contains begin of sample, end is given by sample_point + num_per_set
train_splits = [0,6,12,18]

assert(max(train_splits) <= lim)

for i in range(num_rfs):
    S_lim = [train_splits[i], train_splits[i+1]]
    #S_lim = train_splits
    print("#######################################")
    print("Move to S%d in range [%d,%d]:" %(i, S_lim[0], S_lim[1]) )
    
    training_set = buildFeatures(S_lim, rfs)
    # train random forest with features that we have built
    rfs.append(RandomForestClassifier(n_estimators=200, oob_score=True))
    
    print("Train rf%d"%(i))
    rfs[-1].fit(*training_set)
    print("OOB SCORE",i, rfs[-1].oob_score_)

print("#######################################")
print("Move to Test Section:")

###########################
#  Testint the RFS
# =========================


"""
    Solves multicut problem from trained random forests.
    First: Apply sequence of random forest to test sequence (for each 
    random forest, predict on test image and use output of random forest
    as input of features of the next random forest)
    Secnod: Solve multicut for this 
Args:
    rf: list of trained random forest
    ds: the data set specification, i.e. 'test' or 'train'
    iteration: integer used for saving the multicut result (might use it
                as a feature later on)
    plotting: 'on' or 'off'

Output:
    all_mc_results | first index: number of images, second index: rf
"""

rawDset = rawDsets[ds]
pmapDset = pmapDsets[ds]
dataDset = computedData[ds]

# for each slice
all_mc_results = []
all_predictions = []
print("For Predict and Cut edges using %s images."%(ds))
for z in range(rawDset.shape[0]):   
    print("-- Image", z)
    data = dataDset[z]

    pmap = pmapDset[z,...]
    overseg = data['overseg']
    gt = data['gt']
    img_feats = data['features']
    rag = data['rag']
    raw = rawDset[z,...]

    # get features up to last random forest
    # if we have more than one random forest, we need to calculate 
    # the rf_feats
    mc_results_per_img = []
    predictions_per_img = []
    
    for i, rf in enumerate(rfs):
        print("---- Predicting with rf%d" %(i))
        # initialise with standard features
        features = img_feats
        
        if i > 0:
            print(i)
            rf_feats = feat_from_edge_prob(rag, raw, 
                                           predictions_per_img[-1], 
                                           mc_results_per_img[-1])
            #features = rf_feats
            features = numpy.hstack([rf_feats, img_feats])

        edge_probs = rf.predict_proba(features)[:,1]
        #edge_probs = rf.predict(features)
        predictions_per_img.append(edge_probs)
        MulticutObjective = rag.MulticutObjective
        eps =  0.00001
        p1 = numpy.clip(edge_probs, eps, 1.0 - eps) 
        weights = numpy.log((1.0-p1)/p1)
        objective = MulticutObjective(rag, weights)
        print("---- Solving multicut for rf%d predictions" %(i))
        solver = MulticutObjective.greedyAdditiveFactory().create(objective)
        mc_results = solver.optimize(visitor=MulticutObjective.verboseVisitor())
        mc_results_per_img.append(mc_results)
        # now, neglect last random forest, this is the one we want to
        # do the final predicting on
        result = nifty.graph.rag.projectScalarNodeDataToPixels(rag, mc_results)
    
        # plot for each "multiplier"th slice
        if plotting == 'on' and z % plot_multiplier == 0:
            figure = pylab.figure() 
            figure.suptitle('For RF%d Test Set Results Slice %d'%(i, z), fontsize=20)
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
            pylab.savefig('figs/testing_testRF%d_Img%d.pdf' %(i, z))

    all_mc_results.append(mc_results_per_img)
    all_predictions.append(predictions_per_img)

"""
Input:
    list of segmentation images
Output:
    mean precision and recall values over all images
"""

###########################
# ----- MAIN CODE ---------
#  Benchmarking
# =========================

from nifty import ground_truth

ds = 'test'
rawDset = rawDsets[ds]
dataDset = computedData[ds]
# dimensions: test_images, random_forests, benchmarks
error = numpy.zeros([rawDset.shape[0], num_rfs, 2])
# for each slice
for z in range(rawDset.shape[0]):
        
    data = dataDset[z]
    partialGt = data['partialGt']
    rag = data['rag']
    multicutResults_per_img = all_mc_results[z]
    for r in range(num_rfs):
        
        multicutResults_per_rf = multicutResults_per_img[r]        
        seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, multicutResults_per_rf)    
        randError = ground_truth.RandError(partialGt, seg, ignoreDefaultLabel = True)
        variationError = ground_truth.VariationOfInformation(partialGt, seg, ignoreDefaultLabel = True)
        error[z,r,:] = [randError.error, variationError.value]

mean_error = numpy.mean(error, axis = 0)
print(mean_error)

#z = 1
#data = dataDset[z]
#partialGt = data['partialGt']
#seg = data['overseg']
#randError = ground_truth.RandError(partialGt, seg, ignoreDefaultLabel = True)
#variationError = ground_truth.VariationOfInformation(partialGt, seg, ignoreDefaultLabel = True)
#print(randError.error)
#print(variationError.value)

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