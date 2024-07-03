'''
Video-based Automatic License Plate Recognition (ALPR) system

!!! Visual Rhythm (VR) approach !!!

Usage:
    python vr-alpr.py {optinal arguments}
    --line: line height position                    [defalt:  800]
    --interval: interval between VR images (frames) [default: 600]
    
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



def ALPR(VR, line, cap, ini, double):
    # detect marks
    marks = model_mark(VR, iou=0.05, conf=0.15, verbose=False)
    marks = marks[0].cpu().numpy()
    boxes = marks.boxes.xyxy  

    print(f'Detected {len(boxes)} marks')
    cv2.imwrite(f'results/VR/{ini}.jpg', VR)


    # draw detected bboxs in the VR image
    for box, conf in zip(boxes, marks.boxes.conf):
        x0, y0, x1, y1 = np.floor(box).astype(int)
        cv2.rectangle(VR, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(VR, str(round(conf, 2)), (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(VR, f'{ini}_{ini + y1}.jpg', (x0+90, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imwrite(f'results/VR/{ini}_.jpg', VR)


    # for each mark
    duo = []
    for box in boxes:
        x0_m, y0_m, x1_m, y1_m = np.floor(box).astype(int)
        
        # save coordinates for the next VR
        if y1_m >= len(VR) - 1:
            duo.append([x0_m, x1_m])
        elif y0_m <= 1: # check marks in the previous VR 
            mid = (x1_m + x0_m) // 2
            if any(x1_ant <= mid <= x2_ant for x1_ant, x2_ant in double):
                continue 

        
        # gets the corresponding frame 
        frame_idx = ini + y1_m
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read() 
        if not ret:
            return duo

        # detect plates
        plates = model_plate(img, iou=0.05, conf=0.15, verbose=False)
        plates = plates[0].cpu().numpy()
        

        # find the vehicle associated with the mark
        dist_min = np.shape(img)[0]
        plate = -1
        for xyxy in plates.boxes.xyxy:
            x0, y0, x1, y1 = np.floor(xyxy).astype(int)
            mid = (x0 + x1) // 2
            dist = abs(line - y1)

            if x0_m <= mid <= x1_m and dist < dist_min:
                dist_min = dist
                plate = img[y0:y1 , x0:x1]
    
        
        if np.array_equal(plate, -1):
            print(f"Couldn't find the plate in frame {frame_idx}!!!")
            print('VERIFY IT MANUALLY!')

            # draw detections
            cv2.line(img, (0, line), (np.shape(img)[1], line), (0, 255, 0), 3)
            cv2.line(img, (x0_m, line-50), (x0_m, line+50), (0, 255, 0), 3)
            cv2.line(img, (x1_m, line-50), (x1_m, line+50), (0, 255, 0), 3)
            for xyxy in plates.boxes.xyxy:
                x0, y0, x1, y1 = np.floor(xyxy).astype(int)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)

            cv2.imshow('Frame', cv2.resize(img, (888, 500)))
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # OCR the plate      
            # ...

            cv2.imwrite(f'results/VR/{ini}_{frame_idx}.jpg', plate)

    print()
    return duo

    

def main(video_path, line, interval):
    # opens the video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened() == True, 'Can not open the video :('


    # skip initial frames 
    ini = 120 
    cap.set(cv2.CAP_PROP_POS_FRAMES, ini)


    # initalize an empty VR image
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VR = np.empty((interval, width, 3), dtype=np.uint8)
    frame_idx = 0
    

    # creates folders to save images
    os.makedirs('results/', exist_ok=True)
    os.makedirs('results/VR/', exist_ok=True)


    # reads the video
    double = []  # marks on the border of the previous VR
    while True: 
        ret, frame = cap.read() 
        if not ret:
            break
        
        # stack line in VR image
        VR[frame_idx] = frame[line]
        frame_idx += 1
        
        # if VR image has been fully built
        if frame_idx == interval:
            print(f'VR image {ini} created')
            double = ALPR(VR, line, cap, ini, double)                

            ini += interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, ini) 

            frame_idx = 0       
        

    # last VR image (its size is less than interval)
    if frame_idx > 0:
        print(f'VR image {ini} created')
        double = ALPR(VR[:frame_idx], line, cap, ini, double)      


    # closes the video
    cap.release()
    cv2.destroyAllWindows()
    

    
if __name__ == "__main__":    
    # parse arguments
    parser = argparse.ArgumentParser(description='VR-Based ALPR system')
    
    parser.add_argument('--line', type=int, default=800, help='Line position')
    parser.add_argument('--interval', type=int, default=600, help='Interval between VR images (frames)')
    
    video_path = input('Enter the video path: ')
    args = parser.parse_args()
    

    main(video_path, args.line, args.interval)