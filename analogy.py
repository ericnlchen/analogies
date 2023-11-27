import cv2

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
    #   ex. features_A is a pyramid of images with R, G, B, and feature channels
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
    #   A[l, p_row, p_col, c] where:
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
                B_prime[l, q] = A_prime[l, p]

                # Keep track of the mapping between p and q
                s[l, q] = p

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
def bestApproximateMatch(A, A_prime, B, B_prime, l, q):
    return None

def bestCoherenceMatch(A, A_prime, B, B_prime, s, l, q):
    return None