import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrtools as pt # pip install pyrtools
from annoy import AnnoyIndex # pip install annoy ; https://sds-aau.github.io/M3Port19/portfolio/ann/
import colorsys
from sklearn.decomposition import PCA # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html


def createImageAnalogy(A, A_prime, B, show=False, seed_val=None):
    '''
    Given images A, A_prime and B, returns the image B_prime using
    analogies.
    '''
    if (seed_val is not None):
        np.random.seed(seed_val)

    # Constants
    num_levels = 1 # TODO: make this higher

    print("Initializing data structures...")

    # Create Gaussian pyramids
    pyramid_A = createGaussianPyramid(A, num_levels)
    pyramid_A_prime = createGaussianPyramid(A_prime, num_levels)
    pyramid_B = createGaussianPyramid(B, num_levels)
    pyramid_B_prime = [np.zeros_like(pyramid_B[l]) for l in range(num_levels)]

    # Get features for each level of Gaussian pyramids
    #   ex. A becomes a pyramid of images with R, G, B, and feature channels
    features_A = computeFeatures(pyramid_A)
    features_A_prime = computeFeatures(pyramid_A_prime)
    features_B = computeFeatures(pyramid_B)

    # The s data structure is used to store the pixel
    #   mappings that we find at each level.
    #   It will be the same shape as B' but instead of storing colors it will
    #   store the pixel indices where those colors came from in A'
    s = [np.zeros((pyramid_B_prime[l].shape[0], pyramid_B_prime[l].shape[1]), dtype=object) for l in range(num_levels)]

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
        for q_row in range(pyramid_B_prime[l].shape[0]):
            for q_col in range(pyramid_B_prime[l].shape[1]):
                q = (q_row, q_col)

                if (show):
                    print("Processing pixel at index {}".format(q))
                
                # Find the index p in A and A' which best matches index q in B and B'
                p = bestMatch(features_A, features_A_prime, features_B, pyramid_B_prime, s, l, q)

                # Set the pixel in B' equal to the match we found
                pyramid_B_prime[l][q[0], q[1]] = pyramid_A_prime[l][p[0], p[1]]

                # Keep track of the mapping between p and q
                s[l][q[0], q[1]] = p

    return pyramid_B_prime[0]

def createGaussianPyramid(img, level):
    gaus_pyramid = [img]
    for i in range(level):
        blur_img = cv2.GaussianBlur(gaus_pyramid[i], (5,5), 0)
        downsample_img = blur_img[::2, ::2]
        gaus_pyramid.append(downsample_img)    
    return gaus_pyramid

def bestMatch(A, A_prime, B, B_prime, s, l, q):
    A_l = A[l]
    B_l = B[l]

    P_app = bestApproximateMatch(A, A_prime, B, B_prime, l, q)
    P_coh = bestCoherenceMatch(A, A_prime, B, B_prime, s, l, q)

    if P_coh is None:
        return P_app

    # NOTE:F_l[p] to denote the concatenation of all the feature vectors in neighborhood
    d_app = np.linalg.norm(getFeatureAtQ(A_l, P_app) - getFeatureAtQ(B_l, q))
    d_coh = np.linalg.norm(getFeatureAtQ(A_l, P_coh) - getFeatureAtQ(B_l, q))

    # NOTE: k represents an estimate of the scale of "textons" at level l
    k = 0.7
    # TODO: add the number of levels to this weighting function
    if d_coh <= d_app * (1 + np.power(2, l - 0) * k):
        return P_coh
    else:
        return P_app

# Algorithm: using approximate nearest neighbor search
def bestApproximateMatch(A, A_prime, B, B_prime, l, q):
    '''
    l is for level l
    q is the point inside image B
    '''

    num_samples = 100 # 2000
    patch_size = 5
    A_l = A[l]
    B_l = B[l]

    # TREE is a tuning parameter
    TREE = 10
    _, width, num_features = A_l.shape
    
    t = AnnoyIndex(num_features * patch_size * patch_size, 'euclidean')

    # Randomly sample pixel indices from A
    random_rows = np.random.randint(0, A_l.shape[0] - patch_size, size=num_samples)
    random_cols = np.random.randint(0, A_l.shape[1] - patch_size, size=num_samples)

    i = 0
    for row, col in zip(random_rows, random_cols):
        feature = getFeatureAtQ(A_l, (row + patch_size//2, col + patch_size//2))
        t.add_item(i, feature)
        i += 1
    
    t.build(TREE)

    feature_q = getFeatureAtQ(B_l, (q[0], q[1]))
    
    neighbor_index = t.get_nns_by_vector(feature_q, 1)[0]

    first_pixel_row = random_rows[neighbor_index]
    first_pixel_col = random_cols[neighbor_index]

    center_pixel_row = first_pixel_row + patch_size // 2
    center_pixel_col = first_pixel_col + patch_size // 2

    return (center_pixel_row, center_pixel_col)

def getFeatureAtQ(A, q):
    '''
    Gets the full features for the patch surrounding the pixel q
    (q is in the center of the patch)
    '''
    if (q[0] < 0 or q[0] >= A.shape[0] or
        q[1] < 0 or q[1] >= A.shape[1]):
        return None

    # TODO: use features of A_prime in the feature vector too
    feature_length = A.shape[2]
    patch_size = 5

    q_top_left = (q[0] - patch_size//2, q[1] - patch_size//2)

    # Get a patch
    if q_top_left[0] < 0 or q_top_left[1] < 0 or q_top_left[0]+patch_size >= A.shape[0] or q_top_left[1]+patch_size >= A.shape[1]:
        patch = np.zeros((patch_size, patch_size, feature_length))
        for i in range(patch_size):
            for j in range(patch_size):
                A_i = 0
                A_j = 0
                if q_top_left[0] + i < 0:
                    A_i = 0
                elif q_top_left[0] + i >= A.shape[0]:
                    A_i = A.shape[0] - 1
                else:
                    A_i = q_top_left[0] + i
                if q_top_left[1] + j < 0:
                    A_j = 0
                elif q_top_left[1] + j >= A.shape[1]:
                    A_j = A.shape[1] - 1
                else:
                    A_j = q_top_left[1] + j
                patch[i, j] = A[A_i, A_j]
    else:
        patch = A[q_top_left[0]:q_top_left[0]+patch_size, q_top_left[1]:q_top_left[1]+patch_size, :]

    # Multiply by gaussian kernel to give more weight to center pixels of patch
    kernel_x = cv2.getGaussianKernel(patch_size, sigma=0.3*((patch_size-1)*0.5 - 1) + 0.8)
    kernel_2d = np.outer(kernel_x, kernel_x.T)
    # Normalize the kernel to ensure its values sum up to 1
    kernel_2d /= np.sum(kernel_2d)

    patch = patch * np.stack([kernel_2d] * feature_length, axis=-1)

    # Flatten features
    feature = np.reshape(patch, (-1))

    return feature

def clamp(value, low, high):
    return max(min(value, high), low)
    

def bestCoherenceMatch(A, A_prime, B, B_prime, s, l, q):
    B_prime_l = B_prime[l]
    B_l = B[l]
    A_l = A[l]

    patch_size = 5
    min_row = clamp(q[0] - patch_size//2, 0, B_prime_l.shape[0] - 1)
    max_row = clamp(q[0] + patch_size//2, 0, B_prime_l.shape[0] - 1)
    min_col = clamp(q[1] - patch_size//2, 0, B_prime_l.shape[1] - 1)
    max_col = clamp(q[1] + patch_size//2, 0, B_prime_l.shape[1] - 1)

    r_star = None
    smallest_dist = np.inf
    
    for r_row in range(min_row, max_row + 1):
        for r_col in range(min_col, max_col + 1):
            if (r_row < q[0] or (r_row == q[0] and r_col < q[1])):
                # This is a valid neighbor, do the calculation
                s_r = s[l][r_row, r_col]
                F_s_r = getFeatureAtQ(A_l, (s_r[0] + q[0] - r_row, s_r[1] + q[1] - r_col))
                if F_s_r is None:
                    continue
                F_q = getFeatureAtQ(B_l, q)
                distance = np.linalg.norm(F_s_r - F_q)
                if distance < smallest_dist:
                    r_star = (r_row, r_col)
                    smallest_dist = distance
    if r_star == None:
        return None
    else:
        s_r_star = s[l][r_star[0], r_star[1]]
        p_coh_r = s_r_star[0] + q[0] - r_star[0]
        p_coh_c = s_r_star[1] + q[1] - r_star[1]
        # Check if candidate point is out of bounds
        if (p_coh_r >= A_l.shape[0] or p_coh_r < 0 or
            p_coh_c >= A_l.shape[1] or p_coh_c < 0):
            return None
        return (p_coh_r, p_coh_c)

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

    return feature_pyramid

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