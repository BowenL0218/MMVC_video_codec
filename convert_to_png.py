import cv2
import glob
import os
    
video_path = '/z/mrakeshc/videos/general/'
num = 0
for f in glob.glob(video_path + '/**/*.mp4', recursive=True):
    cap = cv2.VideoCapture(f)
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            save_path = '/z/mrakeshc/clic2022/video_' + str(num)
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(save_path + "/frame%d.png" % count, frame)
            count += 1
        else:
            break
    num += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f)
    
# video_path = '/nfs/turbo/coe-hunseok/mrakeshc/videoSRC30_05.mp4'
# cap = cv2.VideoCapture(video_path)
# count = 0
# while(cap.isOpened()):
#     ret, frame = cap.read()
    
#     if ret:
#         cv2.imwrite("/nfs/turbo/coe-hunseok/mrakeshc/parrots/frame%d.png" % count, frame)
#         count += 1
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()

# video = cv2.VideoCapture(video_path);
#     # Find OpenCV version
# (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
# if int(major_ver)  < 3 :
#     fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
#     print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
# else :
#     fps = video.get(cv2.CAP_PROP_FPS)
#     print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
# video.release()
