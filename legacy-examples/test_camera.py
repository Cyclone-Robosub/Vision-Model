import os
import sys
import argparse
import glob
import cv2

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--source', help='Image or video source', required=True)
parser.add_argument('--resolution', help='Resolution in WxH (e.g., 640x480)', default=None)
args = parser.parse_args()

# Parse arguments
img_source = args.source
user_res = args.resolution

# Determine source type
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Invalid input: {img_source}')
    sys.exit(0)

# Parse resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Load image sources
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type == 'video' or source_type == 'usb':
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if resize:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Display loop
img_count = 0
while True:
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images shown.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret:
            print('End of stream or error.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()

    if resize and source_type != 'picamera':
        frame = cv2.resize(frame, (resW, resH))

    cv2.imshow('Source Display', frame)
    key = cv2.waitKey(0 if source_type in ['image', 'folder'] else 5)

    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.waitKey()

# Cleanup
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
cv2.destroyAllWindows()
