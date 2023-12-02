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
    num_levels = 1 # TODO: make this higher

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

    '''
    NOTE: After this point, an image (A, B, etc) can be indexed as follows:
      A[l][p_row, p_col, c] where:
          - l is the level of the gaussian pyramid
          - p_row is the pixel's row
          - p_col is the pixel's column
          - c is the channel number (including R, G, B, and additional features)
    '''

    # Synthesize B_prime level by level
    # seik: why are we not doing for `l in range(num_levels) - 1, -1, -1` which means we're going from coarest to finest? 
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

    return B_prime[0]

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
    '''   
    NOTE:F_l[p] to denote the concatenation of all the feature vectors in neighborhood
    d_app = (F_l[P_app]-F_l[q])**2
    d_coh =  (F_l[P_coh]-F_l[q])**2
    # NOTE: k represents an estimate of the scale of "textons" at level l
    if d_coh <= d_app * (1 + np.power(2,l - L) * k):
        return P_coh
    else:
        return P_app
    '''
    return P_app

# Algorithm: using approximate nearest neighbor search
def bestApproximateMatch(A, A_prime, B, B_prime, l, q):
    '''
    l is for level l
    q is the point inside image B
    '''
    # TREE is a tuning parameter
    TREE = 10
    _, width, feature_length= A.shape
    
    t = AnnoyIndex(feature_length, 'euclidean')

    # Randomly sample pixel indices from A
    num_samples = 2000
    patch_size = 5

    random_rows = np.random.randint(0, A.shape[0] - patch_size, size=num_samples)
    random_cols = np.random.randint(0, A.shape[1] - patch_size, size=num_samples)

    i = 0
    for row, col in zip(random_rows, random_cols):
        feature = getFeatureAtQ(A, (row, col))
        t.add_item(i, feature)
        i += 1
    
    t.build(TREE)

    feature_q = getFeatureAtQ(B, q)
    
    neighbor_index = t.get_nns_by_vector(feature_q, 1)[0]

    first_pixel_row = random_rows[neighbor_index]
    first_pixel_col = random_cols[neighbor_index]

    center_pixel_row = first_pixel_row + patch_size // 2
    center_pixel_col = first_pixel_col + patch_size // 2

    return (center_pixel_row, center_pixel_col)

def getFeatureAtQ(A, q):

    feature_length = A.shape[2]
    patch_size = 5
    # Get a patch
    patch = A[q[0]:q[0]+patch_size, q[1]:q[1]+patch_size, :]

    # Multiply by gaussian kernel to give more weight to center pixels of patch
    kernel = cv2.getGaussianKernel(patch_size) # default sigma = 0.3*((patch_size-1)*0.5 - 1) + 0.8
    patch = patch * np.stack([kernel] * feature_length, axis=-1)

    # Flatten features
    feature = np.reshape(patch, (-1))

    return feature

    

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
    num_features = 13 # luminance + 12 steerable pyramid responses

    feature_pyramid = [np.zeros((pyramid[l].shape[0], pyramid[l].shape[1], num_features)) for l in range(num_levels)]

    # For each level of the pyramid...
    # Steerable pyramid library etc: https://github.com/LabForComputationalVision/pyPyrTools
    for l in range(num_levels):
        feature_pyramid[l][:, :, 0] = computeLuminance(pyramid[l])
        feature_pyramid[l][:, :, 1:] = computeSteerablePyramidResponse(pyramid[l])

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
    pyr = pt.pyramids.SteerablePyramidSpace(im, height=3, order=3)

    # Get target size (original size of full scale image)
    target_shape = (pyr.pyr_coeffs[(0, 0)].shape[1],
                    pyr.pyr_coeffs[(0, 0)].shape[0])

    # Put the steerable pyramid response in array format
    responses = []
    for key, response in pyr.pyr_coeffs.items():
        if (type(key) == tuple):
            # Resize to match the original image size
            response_resized = cv2.resize(response, target_shape, interpolation=cv2.INTER_NEAREST)
            # Add to the stack of responses
            responses.append(response_resized)
    result = np.stack(responses, axis=-1)
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