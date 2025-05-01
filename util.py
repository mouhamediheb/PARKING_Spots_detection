
import cv2
from skimage.io import imread
from skimage.transform import resize
import pickle

def get_Parking_spot_bboxes(connected_components):
    (total_labels,id_labels,stats,controides)=connected_components
    slots=[]
    coef=1
    for i in range(1,total_labels):
        x1 = int(stats[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(stats[i, cv2.CC_STAT_TOP] * coef)
        w = int(stats[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(stats[i, cv2.CC_STAT_HEIGHT] * coef)
        slots.append([x1,y1,w,h])
    return slots
EMPTY=True
NOT_EMPTY=False
# Load the model from the file
with open('model.p', 'rb') as f:
        MODEL = pickle.load(f)


def empty_or_notempty(bgr_img):
    flatten_data = []
    img = resize(bgr_img, (15, 15, 3))
    flatten_data.append(img.flatten())  # only one image
    y_output = MODEL.predict(flatten_data)

    if y_output==0:
        return EMPTY
    else :
        return NOT_EMPTY
