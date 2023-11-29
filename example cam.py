'''
This test runs stella_vslam on camera, 640x480 px.
Opens Pangolin.
You need to provided the configuration files belonging to your calibrated camera.
'''

import stellavslam
import cv2 as cv
import sys
import argparse
from threading import Thread

# Some arguments from run_video_slam.cc
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vocab", help="vocabulary file path", default="./orb_vocab.fbow")
parser.add_argument("-m", "--video", help="video file path", default="0")
parser.add_argument("-c", "--config", help="config file path", default="./vslam/config Logitech c270 640x480 calibrado.yaml")
parser.add_argument("-p", "--map_db", help="store a map database at this path after SLAM")
#parser.add_argument("-f", "--factor", help="scale factor to show video in window - doesn't affect stella_vslam", default=0.5, type=float)
args = parser.parse_args()

def run_slam():
    #global frameShowFactor
    global video
    global SLAM

    pose = []
    timestamp = 0.0
    print("Entering the video feed loop.")
    print("You should soon see the video in a window, and the 4x4 pose matrix on this terminal.")
    print("ESC to quit (focus on window: click on feeding frame window, then press ESC).")
    is_not_end = True   
    
    while(is_not_end):
        is_not_end, frame = video.read()    
        if(frame.size):
            retVal, pose = SLAM.feed_monocular_frame(frame, timestamp) # fake timestamp to keep it simple
        if((timestamp % 30) == 0):
            print("Timestamp", timestamp, ", Pose:")
            
            # Format pose matrix with only a few decimals
            for row in pose:
                for data in row:
                    sys.stdout.write('{:9.1f}'.format(data))
                print()
        timestamp += 1
        key = cv.waitKey(1)  # Needed to refresh imshow window
        if (key == 27):
            # ESC, finish
            break


config = stellavslam.config(config_file_path=args.config)
SLAM = stellavslam.system(cfg=config, vocab_file_path=args.vocab)
VIEWER = stellavslam.viewer(config, SLAM)
SLAM.startup()
print("stellavslam up and operational.")

if args.video.isnumeric():
    args.video = int(args.video)

video = cv.VideoCapture(args.video)
video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 480)


#frameShowFactor = args.factor

slamThreadInstance = Thread(target=run_slam)
slamThreadInstance.start()
VIEWER.run()
slamThreadInstance.join()
SLAM.shutdown()