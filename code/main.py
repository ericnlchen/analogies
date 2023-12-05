'''
Project: Texture - main.py
CSCI 1290 Computational Photography, Brown U.
Written by Emanuel Zgraggen.
Converted to Python by Trevor Houchens.


Usage-

To run the texture synthesis section:
    
    python main.py synthesis
    python main.py synthesis -q

To run the texture transfer section:

    python main.py transfer -s <input_file> -t <transfersource_file>

    input_file is the file that you want to restyle
    transfersource_file is the texture used to restyle the input
    Only the filename is needed, not the full path (i.e. 'toast.png' not '~/<dir1>/<dir2>/toast.png')

'''
'''
import os
import numpy as np
import cv2
from analogy import createGaussianPyramid, computeFeatures, getFeatureAtQ, createImageAnalogy
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm

SOURCE_PATH = '../data'
OUTPUT_PATH = '../output/'
SYNTHESIS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'synthesis')
TRANSFER_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'transfer')

def synthesis(args):
    # List the full file names of each of the input images
    input_list = [os.path.join(SOURCE_PATH, 'textures', img) for img in os.listdir(os.path.join(SOURCE_PATH, 'textures'))\
                 if img.endswith('.jpg') or img.endswith('.bmp') or img.endswith('.png')]
    # Make output directory
    # if not os.path.exists(SYNTHESIS_OUTPUT_PATH):
    #     os.mkdir(SYNTHESIS_OUTPUT_PATH)
    os.makedirs(SYNTHESIS_OUTPUT_PATH, exist_ok=True)

    for i, path in enumerate(input_list):

        print(f"---------- Image {i}, {os.path.basename(path)} ----------\n")

        img = cv2.imread(input_list[i])
        img = img.astype(np.float32) / 255.

        # specify output size and algorithm parameters
        outsize = (320, 480)
        tilesize = 60
        overlapsize = 10

        base = os.path.basename(path).split(".")[0]

        #  method => 1 = random, 2 = best ssd, 3 = best ssd + min cut

        # random patch
        # out = image_synthesis(img, outsize, tilesize, overlapsize, 1, args["quiet"])
        # cv2.imwrite(os.path.join(SYNTHESIS_OUTPUT_PATH, f'{base}_random.png'), out*255)
        
        # best ssd   
        # TODO: Uncomment this when you have implemented ssd tile selection
        # out = image_synthesis(img, outsize, tilesize, overlapsize, 2, args["quiet"])
        # cv2.imwrite(os.path.join(SYNTHESIS_OUTPUT_PATH, f'{base}_ssd.png'), out*255)
        
        # best ssd + min cut   
        # TODO: Uncomment this when you have implemented ssd + minimum error boundary cut tile selection
        out = image_synthesis(img, outsize, tilesize, overlapsize, 3, args["quiet"])
        cv2.imwrite(os.path.join(SYNTHESIS_OUTPUT_PATH, f'{base}_min_cut.png'), out*255)


def transfer(args):

    if args['input'] is None or args['transfersource'] is None:
        print("You must provide an input image and a transfer source image for texture transfer")

    else:
        # Make output directory
        if not os.path.exists(TRANSFER_OUTPUT_PATH):
            os.mkdir(TRANSFER_OUTPUT_PATH)

        # read images
        input = cv2.imread(os.path.join(SOURCE_PATH, "images", args['input']))
        transfersource = cv2.imread(os.path.join(SOURCE_PATH, "textures", args['transfersource']))
        # convert to float in [0,1] range
        input = input.astype(np.float32) / 255.
        transfersource = transfersource.astype(np.float32) / 255.

        print(f"Reproducing {args['input']} with the texture of {args['transfersource']}")

        # hyperparameters to define patch (tile) size, overlap region, and number of iterations for passes.
        tilesize=36
        overlapsize=6
        n_iterations=3
        outsize = (input.shape[0], input.shape[1])

        output = image_texture(input, transfersource, outsize, tilesize, overlapsize, n_iterations, args["quiet"])
        
        # write out
        input_name = args['input'].split(".")[0]
        transfersource_name = args['transfersource'].split(".")[0]
        cv2.imwrite(os.path.join(TRANSFER_OUTPUT_PATH, f"{input_name}_{transfersource_name}_transfer.jpg"), output*255)
'''
# if __name__ == "__main__":
    # func_map = {'transfer': transfer, 'synthesis': synthesis}

    # parser = argparse.ArgumentParser(description="CSCI1290 - Project Texture")
    # parser.add_argument("method", help="Name of the method to run ('transfer' or 'synthesis')")
    # parser.add_argument("-i", "--input", help="Name of the input image with extension. Used for method 'synthesis', and is the image to texturize in method 'transfer'.")
    # parser.add_argument("-t", "--transfersource", help="Name of the texture image with extension to be used as the source of the texture in method 'transfer'")
    # parser.add_argument("-q", "--quiet", help="used to stop displaying images", action="store_true")
    # args = vars(parser.parse_args())

    # if args["method"] in func_map:
    #     func_map[args["method"]](args)
    # else:
    #     print(f"{args['method']} is not a supported command. Try using 'synthesis' or 'transfer'")
    # A = plt.imread('../data/big-orange.jpeg')
    # pyramid_A = createGaussianPyramid(A, 5)
    # A = computeFeatures(pyramid_A)
    # feature_vector = getFeatureAtQ(A[0], (0, 0))
    # print(feature_vector.shape)

import matplotlib.pyplot as plt
import cv2
from analogy import createImageAnalogy

if __name__ == '__main__':
    
    A = plt.imread('../data/a1.jpg')
    A_prime = plt.imread('../data/a2.jpg')
    # A_prime = cv2.GaussianBlur(A, (15,15), 10)
    # plt.imshow(A_prime)
    # plt.show()
    B = plt.imread('../data/sky2.jpg')
    B_prime = createImageAnalogy(A, A_prime, B, show=True, seed_val=0)
    plt.imshow(B_prime)
    plt.show()
    plt.imsave("../results/output.jpg", B_prime)
