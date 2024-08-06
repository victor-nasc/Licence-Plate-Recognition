'''
Video-based Automatic License Plate Recognition (ALPR) system

!!! Visual Rhythm (VR) approach !!!

Usage:
    python vr-alpr.py {optinal arguments}
    --line_height: line height position             [defalt:  1000]
    --interval: interval between VR images (frames) [default: 900]
    
    The video path is prompted during execution.
'''


import cv2
import numpy as np
from ultralytics import YOLO

import argparse
import os


# load models
model_mark = YOLO('./YOLO/mark.pt')
model_plate = YOLO('./YOLO/plate.pt')


def ALPR(VR, line_height, cap, time):
    # detect marks
    marks = model_mark(VR, iou=0.05, conf=0.15, verbose=False)
    marks = marks[0].cpu().numpy()
    boxes = marks.boxes.xyxy  

    print()
    print(f'Detected {len(boxes)} marks')
    cv2.imwrite(f'results/VR/{time}.jpg', VR)

    # draw detected bboxs in the VR image
    for box, conf in zip(boxes, marks.boxes.conf):
        x0, y0, x1, y1 = np.floor(box).astype(int)
        cv2.rectangle(VR, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(VR, str(round(conf, 2)), (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(VR, f'{time}_{time + y1}.jpg', (x0+90, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imwrite(f'results/VR/{time}_.jpg', VR)

    # for each mark
    for box in boxes:
        l, _, r, y1 = np.floor(box).astype(int)
        l -= 10
        r += 10

        # mark in the top of the VR image
        if y1 >= len(VR) - 1:
            continue # skip it. In the next VR image we will analyze it
        
        # gets the corresponding frame 
        frame_idx = time + y1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx-5)
        _, frame = cap.read() 

        # detect plates
        plates = model_plate(frame, iou=0.05, conf=0.15, verbose=False)
        plates = plates[0].cpu().numpy()
        
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
            print(f"Couldn't find the plate in frame {frame_idx}!!!")
            print('VERIFY IT MANUALLY!')

            # draw detections
            cv2.line(frame, (0, line_height), (np.shape(frame)[1], line_height), (0, 255, 0), 3)
            cv2.line(frame, (l, line_height-50), (l, line_height+50), (0, 255, 0), 3)
            cv2.line(frame, (r, line_height-50), (r, line_height+50), (0, 255, 0), 3)
            for xyxy in plates.boxes.xyxy:
                x0, y0, x1, y1 = np.floor(xyxy).astype(int)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 3)

            cv2.imshow(f'{time}_{frame_idx}', cv2.resize(frame, (888, 500)))
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # OCR the plate      
            # ...

            cv2.imwrite(f'results/VR/{time}_{frame_idx}.jpg', plate)

    
def main(video_path, line_height, interval):
    # opens the video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened() == True, 'Can not open the video :('

    # skip initial frames 
    time = 120 
    cap.set(cv2.CAP_PROP_POS_FRAMES, time)

    # creates folders to save images
    os.makedirs('results/', exist_ok=True)
    os.makedirs('results/VR/', exist_ok=True)

    # initalize an empty VR image
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VR = np.empty((interval, width, 3), dtype=np.uint8)
    i = 0

    # reads the video
    while True: 
        ret, frame = cap.read() 
        if not ret:
            break
        
        # stack line in VR image
        VR[i] = frame[line_height]
        i += 1
        
        # if VR image has been fully built
        if i == interval:
            print(f'VR image {time} created')
            ALPR(VR, line_height, cap, time)                

            time += interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, time) 
            i = 0       
        
    # last VR image (its size is less than interval)
    if i > 0:
        print(f'VR image {time} created')
        ALPR(VR[:i], line_height, cap, time)      

    # closes the video
    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":    
    # parse arguments
    parser = argparse.ArgumentParser(description='VR-Based ALPR system')
    
    parser.add_argument('--line_height', type=int, default=800, help='Line height position')
    parser.add_argument('--interval', type=int, default=600, help='Interval between VR images (frames)')
    
    video_path = input('Enter the video path: ')
    args = parser.parse_args()

    main(video_path, args.line_height, args.interval)
