'''
Video-based Automatic License Plate Recognition (ALPR) system

!!! Accumulative Line Analysis Algorithm (ALA) approach !!!

Usage:
    python ala-alpr.py {optinal arguments}
    --line_height: line height position                      [defalt:  1000]
    --gamma: threshold to discards clusters and remove noise [default: 100]
    --hide: hide the video from being shown realtime         [default: True]

    The video path is prompted during execution.
'''




import cv2
import numpy as np
from ultralytics import YOLO

import argparse
import os


# load model
model_plate = YOLO('./YOLO/plate.pt')


def ALPR(line_height, frame, l, r, time):
    # detect plates
    plates = model_plate(frame, iou=0.05, conf=0.15, verbose=False)
    plates = plates[0].numpy()

    # find the vehicle associated with the mark
    dist_min = np.shape(frame)[0]
    plate = -1
    for xyxy in plates.boxes.xyxy:
        x0, y0, x1, y1 = np.floor(xyxy).astype(int)
        mid_y = (y1 + y0) // 2
        dist = abs(line_height - mid_y)

        if (l <= x0 <= r or l <= x1 <= r) and dist < dist_min:
            dist_min = dist
            plate = frame[y0:y1 , x0:x1]
                
    # if no plate was detected
    if np.array_equal(plate, -1):
        print(f"Couldn't find the plate in frame {time}!!!")
        print('VERIFY IT MANUALLY!')

        # draw detections
        cv2.putText(frame, 'VERIFY IT MANUALLY!', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
        cv2.line(frame, (0, line_height), (np.shape(frame)[1], line_height), (0, 255, 0), 3)
        cv2.line(frame, (l, line_height-50), (l, line_height+50), (0, 255, 0), 3)
        cv2.line(frame, (r, line_height-50), (r, line_height+50), (0, 255, 0), 3)
        for xyxy in plates.boxes.xyxy:
            x0, y0, x1, y1 = np.floor(xyxy).astype(int)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 3)

        cv2.imshow('VERIFY IT MANUALLY', cv2.resize(frame, (888, 500)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # OCR the plate      
            # ...

        cv2.imwrite(f'results/ALA/{time}.jpg', plate)
    
        
def getClusters(vector):
    vector = vector.flatten()
    
    # indices that change from 0 <--> 1
    changes = np.where(np.diff(vector) != 0)[0] + 1
    
    if vector[-1] == 1:
        changes = np.append(changes, len(vector))
    if vector[0] == 1:
        changes = np.append(0, changes)

    indices = changes.reshape(-1, 2)
    indices[:, 1] -= 1 

    return indices


def main(video_path, line_height, gamma, hide):
    # opens the video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened() == True, 'Can not open the video :('

    # skip initial frames 
    time = 120
    cap.set(cv2.CAP_PROP_POS_FRAMES, time)

    # creates folders to save images
    os.makedirs('results/', exist_ok=True)
    os.makedirs('results/ALA/', exist_ok=True)

    # background subtractior
    bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=1)

    # initialize line_or
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    line_or = np.zeros(width)

    # begins video 
    while True: 
        ret, frame = cap.read() 
        if not ret:
            break
        
        line = frame[line_height]

        # line background subtraction
        line = bg.apply(line)
        line = cv2.blur(line, (5,5))
        _, line = cv2.threshold(line, 50, 255, cv2.THRESH_BINARY)
        line = np.dot(line,[0.114, 0.587, 0.299]) 

        # accumulative line
        line_or = np.logical_or(line_or, line)
        clusters = getClusters(line_or)
        
        # for each cluster
        for l, r in clusters:
            line_xor = np.logical_xor(line[l:r+1], line_or[l:r+1])
            
            if r-l+1 < gamma: 
                line_or[l:r+1] = False
            elif np.sum(line_xor) == r-l+1:
                ALPR(line_height, frame, l, r, time)
                line_or[l:r+1] = False
                
        if not hide:
            # shows the video
            cv2.line(frame, (0, line_height), (width, line_height), (0, 255, 0), 2)
            cv2.imshow('frame', cv2.resize(frame, (888, 500)))
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        time += 1
        
    # closes the video
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":    
    # parse arguments
    parser = argparse.ArgumentParser(description='ALA-Based ALPR system')
    
    parser.add_argument('--line_height', type=int, default=1000, help='Line height position')
    parser.add_argument('--gamma', type=int, default=100, help='Threshold to discards clusters and remove noise')
    parser.add_argument('--hide', action='store_true', help='Hide the video from being shown realtime')

    video_path = input('Enter the video path: ')
    args = parser.parse_args()

    main(video_path, args.line_height, args.gamma, args.hide)
