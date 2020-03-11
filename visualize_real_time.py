import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
import pafy,joblib,cv2
from skimage import color
import Sliding as sd
size = (64,128)
step_size = (10,10)
downscale = 1.25
model = joblib.load('models/models.dat')

# real time person detection 

url = 'https://youtu.be/NyLF8nHIquM'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")
cap = cv2.VideoCapture(play.url)

while True:
    ret,frame = cap.read()
    image = cv2.resize(frame,(512,512))
    detections = []
    #The current scale of the image 
    scale = 0
    for im_scaled in pyramid_gaussian(image, downscale = downscale):
        #The list contains detections at the current scale
        if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
            break
        for (x, y, window) in sd.sliding_window(im_scaled, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue
            window = color.rgb2gray(window)
            fd=hog(window, orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
            fd = fd.reshape(1, -1)
            pred = model.predict(fd)

            if pred == 1:
                
                if model.decision_function(fd) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fd), 
                    int(size[0] * (downscale**scale)),
                    int(size[1] * (downscale**scale))))
        scale += 1
    clone = image.copy()
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print('sc:',sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)

    for(x1, y1, x2, y2) in pick:
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(clone,'Person : {:.2f}'.format(np.max(sc)),(x1-2,y1-2),1,1,(0,122,12),1)
    cv2.imshow('Person Detection',clone)
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()

                
    


