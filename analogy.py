

def create_image_analogy(A, A_prime, B):
    '''
    Given images A, A_prime and B, returns the image B_prime using
    analogies.
    '''
    # Constants
    num_levels = 5    

    # Create Gaussian pyramids
    pyramid_A = create_gaussian_pyramid(A, num_levels)
    pyramid_A_prime = create_gaussian_pyramid(A_prime, num_levels)
    pyramid_B = create_gaussian_pyramid(B, num_levels)

    # Get features for each level of Gaussian pyramids
    #   ex. features_A is a pyramid of images with R, G, B, and feature channels
    A = compute_features(pyramid_A)
    A_prime = compute_features(pyramid_A_prime)
    B = compute_features(pyramid_B)
    B_prime = np.zeros_like(B)

    # The s data structure is used to store the pixel
    #   mappings that we find at each level
    s = [dict() for x in ]

    # NOTE: After this point, an image (A, B, etc) can be indexed as follows:
    #   A[l, p_row, p_col, c] where:
    #       - l is the level of the gaussian pyramid
    #       - p_row is the pixel's row
    #       - p_col is the pixel's column
    #       - c is the channel number (including R, G, B, and additional features)

    # Synthesize B_prime level by level
    for i in range(num_levels):
        # Find the index p in A and A' which best matches index q in B and B'
        p = best_match(A, A_prime, B, B_prime, s, l, q)

        # Set the pixel in B' equal to the match we found
        B_prime[l, q] = A_prime[l, p]

        # Keep track of the mapping between p and q
        s[l, q] = p

    B_prime = None
    return B_prime

def create_gaussian_pyramid(img, le)el:

    gaus_pyramid_ary = [im]    return None

def best_match(A, A_prime, B, B_prime, s, l, q):
    P_app = BestApproximateMatch(A, A_prime, B, B_prime, l, q)
    P_coh = BestCoherenceMatch(A, A_prime, B,prime, s, l, q)
    # NOTE:F_l[p] to denote the concatenation of all the feature vectors in neighborhood
    d_app = (F_l[P_app]-F_l[q])**2
    d_coh =  (F_l[P_coh]-F_l[q])**2
    g