'''
Final Project: Image and Video Analogy
CSCI 1290 Computational Photography, Brown U.
Written by Eric Chen, Seik Oh, Ziyan Liu, 2021
'''

import matplotlib.pyplot as plt
import cv2
from analogy import createImageAnalogy
import os
from tqdm import tqdm

def extract_frames(video_path, frames_dir):
    """
    Extracts frames from a video file and saves them as images.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return 0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print("Error: Failed to open video file.")
        return 0

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Warning: End of video or failed to read a frame.")
            break
        
        
        height, width = frame.shape[:2]
        new_height = 480

        
        scaling_factor = new_height / height
        new_width = int(width * scaling_factor)

        
        resized_image = cv2.resize(frame, (new_width, new_height))
    
    
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f"frame{frame_count:06d}.jpg")
        
        cv2.imwrite(frame_path, resized_image)
        frame_count += 1

    cap.release()
    return frame_count, fps


def create_video(frames_dir, output_video_path, fps):
    """
    Creates a video from a series of frames.
    """
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if not frame_files:
        return

    frame = cv2.imread(frame_files[0])
    height, width, _ = frame.shape
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, cv2_fourcc, fps, (width, height))

    for frame_file in frame_files:
        video.write(cv2.imread(frame_file))

    video.release()
    

def create_video_from_selected_frames(frames_dir, modified_frames_dir, output_video_path, fps):
    """
    Creates a video from a series of selected frames.
    """
    all_frame_files = sorted(os.listdir(frames_dir))
    modified_frame_files = set(os.listdir(modified_frames_dir))

    frame_files = [f for f in all_frame_files if f in modified_frame_files]
    if not frame_files:
        return

    frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = frame.shape
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, cv2_fourcc, fps, (width, height))

    for frame_file in frame_files:
        video.write(cv2.imread(os.path.join(modified_frames_dir, frame_file)))

    video.release()
    

if __name__ == '__main__':
    
    A = plt.imread('../data/eld1.jpg')
    A_prime = plt.imread('../data/eld2.jpg')
    B = plt.imread('../data/shore.jpg')
    B_prime = createImageAnalogy(A, A_prime, B, show=True, seed_val=0)
    # plt.imshow(B_prime)
    # plt.show()
    plt.imsave("../results/output.jpg", B_prime)

    #current sample video reference: https://www.istockphoto.com/video/flock-of-sheep-looking-for-food-on-the-dried-lake-bed-gm1426683353-470839023
    video_path = '../data/captain.MOV'
    frames_dir = '../data/frames'
    modified_frames_dir = '../data/modified_frames'
    results_dir = '../results'  # Directory for results
    output_video_path = os.path.join(results_dir, 'output_video.mp4')
    
    # Read the reference images for analogy
    A = plt.imread('./data/sketchA.jpeg')
    A_prime = plt.imread('./data/sketchA_prime.jpeg')


    #Ensure directories exist
    #ChatGPT used for some debugging for the try-except exception handling
    try:
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(modified_frames_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
        exit(1)

    # Extract frames from the video
    count,fps = extract_frames(video_path, frames_dir)
    if count == 0:
        exit(1)
    
# # # Process only every n-th frame
# # n = 7
# # for frame_number, frame_file in tqdm(enumerate(sorted(os.listdir(frames_dir)))):
# #     if frame_number % n == 0:  # Process only if frame number is a multiple of 12
# #         frame_path = os.path.join(frames_dir, frame_file)
# #         modified_frame_path = os.path.join(modified_frames_dir, frame_file)
        
# #         B = plt.imread(frame_path)
# #         B_prime = createImageAnalogy(A, A_prime, B, show=False, seed_val=0)
# #         plt.imsave(modified_frame_path, B_prime)

# #     # Create a video from the modified frames
# #     create_video_from_selected_frames(frames_dir, modified_frames_dir, output_video_path, fps)

    
    for frame_file in tqdm(sorted(os.listdir(frames_dir))):
        frame_path = os.path.join(frames_dir, frame_file)
        modified_frame_path = os.path.join(modified_frames_dir, frame_file)
        
        B = plt.imread(frame_path)
        B_prime = createImageAnalogy(A, A_prime, B, show=False, seed_val=0)
        plt.imsave(modified_frame_path, B_prime)

    # Create a video from the modified frames
    create_video(modified_frames_dir, output_video_path, fps)  # Adjust FPS as needed
