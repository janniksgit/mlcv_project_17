# figure out how to do other watershed

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

import vigra

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
plotting = 'on'
# debugging: only use part of the training images

# we use part of the training images as ground truth    
split = 29
lim = 1

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
        # filter raw image
        filtered_raw = vigra.filters.hessianOfGaussianEigenvalues(raw.astype('float32')/255, 2)
        # select first eigenvalue
        filtered_raw = filtered_raw[...,0]
        
        # node weighted watershed with local minima seeds
        overseg = nifty.segmentation.seededWatersheds(filtered_raw, method='node_weighted')
        
        # make minimal label = 0
        overseg -= 1
        data['overseg'] = overseg

        # region adjacency graph
        rag = nifty.graph.rag.gridRag(overseg)
        data['rag'] = rag

        # compute features
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




