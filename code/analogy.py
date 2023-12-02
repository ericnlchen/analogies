import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrtools as pt # pip install pyrtools
from annoy import AnnoyIndex # pip install annoy ; https://sds-aau.github.io/M3Port19/portfolio/ann/
import colorsys
from sklearn.decomposition import PCA # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html


def createImageAnalogy(A, A_prime, B):
    '''
    Given images A, A_prime and B, returns the image B_prime using
    analogies.
    '''
    # Constants
    num_levels = 5

    # Create Gaussian pyramids
    pyramid_A = createGaussianPyramid(A, num_levels)
    pyramid_A_prime = createGaussianPyramid(A_prime, num_levels)
    pyramid_B = createGaussianPyramid(B, num_levels)

    # Get features for each level of Gaussian pyramids
    #   ex. A becomes a pyramid of images with R, G, B, and feature channels
    A = computeFeatures(pyramid_A)
    A_prime = computeFeatures(pyramid_A_prime)
    B = computeFeatures(pyramid_B)
    B_prime = np.zeros_like(B)

    # The s data structure is used to store the pixel
    #   mappings that we find at each level.
    #   It will be the same shape as B' but instead of storing colors it will
    #   store the pixel indices where those colors came from in A'
    s = np.zeros(A_prime.shape)

    # NOTE: After this point, an image (A, B, etc) can be indexed as follows:
    #   A[l][p_row, p_col, c] where:
    #       - l is the level of the gaussian pyramid
    #       - p_row is the pixel's row
    #       - p_col is the pixel's column
    #       - c is the channel number (including R, G, B, and additional features)

    # Synthesize B_prime level by level
    for l in range(num_levels):
        # For each pixel q at (q_row, q_col)...
        for q_row in range(B_prime[l].shape[0]):
            for q_col in range(B_prime[l].shape[1]):
                q = (q_row, q_col)
                
                # Find the index p in A and A' which best matches index q in B and B'
                p = bestMatch(A, A_prime, B, B_prime, s, l, q)

                # Set the pixel in B' equal to the match we found
                B_prime[l][q] = A_prime[l][p]

                # Keep track of the mapping between p and q
                s[l][q] = p

    return B_prime[num_levels-1]

def createGaussianPyramid(img, level):
    gaus_pyramid = [img]
    for i in range(level):
        blur_img = cv2.GaussianBlur(gaus_pyramid[i], (5,5), 0)
        downsample_img = blur_img[::2, ::2]
        gaus_pyramid.append(downsample_img)    
    return gaus_pyramid

def bestMatch(A, A_prime, B, B_prime, s, l, q):
    P_app = bestApproximateMatch(A, A_prime, B, B_prime, l, q)
    P_coh = bestCoherenceMatch(A, A_prime, B,B_prime, s, l, q)
    # NOTE:F_l[p] to denote the concatenation of all the feature vectors in neighborhood
    d_app = (F_l[P_app]-F_l[q])**2
    d_coh =  (F_l[P_coh]-F_l[q])**2
    # NOTE: k represents an estimate of the scale of "textons" at level l
    if d_coh <= d_app * (1 + np.power(2,l - L) * k):
        return P_coh
    else:
        return P_app

# Algorithm: using approximate nearest neighbor search
def bestApproximateMatch(A, A_prime, B, B_prime, l, q, features_A, feature_length):
    '''
    l is for level l
    q is the point inside image B
    '''
    # skeloton code for bestApproximateMatch

    #TREE is a tuning parameter
    TREE = 10
    _,width,_ = A.shape[:-1]
    
    t = AnnoyIndex(feature_length, 'euclidean')

    
    for i, feature in enumerate(features_A):
        t.add_item(i, feature)

    t.build(TREE)

    
    feature_q = getFeatureAtQ(B, q)

    
    neighbor_index = t.get_nns_by_vector(feature_q, 1)[0]

    row = neighbor_index // width
    col = neighbor_index % width

    return (row,col)



def bestCoherenceMatch(A, A_prime, B, B_prime, s, l, q):
    N_q = define_the_neighborhood(B_prime, q, l) # some kind of function to determine the neighborhood
    
    r_star = None
    min_diff = float('inf')
    
    for r in N_q:
        # ||F_l(s(r)+(q−r))−F_l(q)||^2
        cur_diff = np.linalg.norm(feature_vector(A, A_prime, B, B_prime, s, r, q, l) - getFeatureAtQ(B,q))
        
        # update the best pixel
        if cur_diff < min_diff:
            min_diff = cur_diff
            r_star = r
    # calcluate best coherent match
    return s[r_star] + (q - r_star)
    


def computeFeatures(pyramid):
    '''
    Given a pyramid of images, returns a pyramid of images with
    R, G, B, and feature channels.
    The input is a list of numpy arrays of shape (numRows x numColumns x 3).
    Output is a list of numpy arrays of shape (numRows x numColumns x numFeatures)

    R, G, and B could be included or not included in the features
    '''
    # Constants
    num_levels = len(pyramid)
    num_features = 17

    feature_pyramid = [np.zeros((pyramid[l].shape[0], pyramid[l].shape[1], num_features)) for l in range(num_levels)]

    # For each level of the pyramid...
    # Steerable pyramid library etc: https://github.com/LabForComputationalVision/pyPyrTools
    for l in range(num_levels):
        feature_pyramid[l][:, :, 0] = computeLuminance(pyramid[l])
        feature_pyramid[l][:, :, 1:] = computeSteerablePyramidResponse(pyramid[l])
        print(feature_pyramid[l].shape)

def computeLuminance(im_BGR):
    '''
    Returns the Y channel from YIQ representation of the image
    '''
    # TODO: use YIQ
    return cv2.cvtColor(im_BGR, cv2.COLOR_BGR2GRAY)

def computeSteerablePyramidResponse(im):
    # Use the grayscale image as input
    im = computeLuminance(im)

    # Apply the steerable pyramid
    filt = 'sp3_filters'
    pyr = pt.pyramids.SteerablePyramidSpace(im, height=4, order=3)

    # Get target size (original size of full scale image)
    target_shape = pyr.pyr_coeffs[(0, 0)].shape

    # Put the steerable pyramid response in array format
    responses = []
    for key, response in pyr.pyr_coeffs.items():
        if (type(key) == tuple):
            # Resize to match the original image size
            response_resized = cv2.resize(response, target_shape, interpolation=cv2.INTER_NEAREST)
            # Add to the stack of responses
            responses.append(response_resized)
    result = np.stack(responses, axis=-1)
    print(result.shape)
    return result

# won't be using this probably -> need to use steerable pyramids
def edge_detection(image_path, low_threshold=100, high_threshold=200):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if image is loaded
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Apply Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold)
    
    return edges