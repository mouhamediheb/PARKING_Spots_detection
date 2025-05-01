import os

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import  train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
#PREPROCESS IMAGES DATA
path_data = 'clf-data'
categories = ['empty', 'not_empty']
data = []
labels = []

for category_idx, category in enumerate(os.listdir(path_data)):
    category_path = os.path.join(path_data, category)
    for file in os.listdir(category_path):
        file_path = os.path.join(category_path, file)
        img = imread(file_path)
        img = resize(img, (15, 15, 3))
        data.append(img.flatten())
        labels.append(category_idx)
data=np.asarray(data)
labels=np.asarray(labels)

#SPLIT DATA TO TRAIN\TEST

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#train_model

classifier=SVC()
parameters={"gamma":[0.01,0.001,0.0001],"C":[1,10,100,1000]}
grid_search=GridSearchCV(classifier,parameters)
grid_search.fit(x_train,y_train)



#test_model
model_clasifier=grid_search.best_estimator_
y_pred=model_clasifier.predict(x_test)
print("{}% of accuracy_score for this model ".format(accuracy_score(y_pred,y_test)*100))


# `grid_search` is your trained model (could also be `model`, `classifier`, etc.)
with open('model.p', 'wb') as f:
    pickle.dump(model_clasifier, f)  # ✅ this is where `.dump()` is used
