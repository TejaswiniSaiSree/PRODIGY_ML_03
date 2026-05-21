import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = []
labels = []

folders = ["Cat", "Dog"]

for label, folder in enumerate(folders):

    path = os.path.join("PetImages", folder)

    count = 0

    for img in os.listdir(path):

        if count >= 500:
            break

        img_path = os.path.join(path, img)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image,(64,64))

        data.append(image.flatten())

        labels.append(label)

        count += 1

X = np.array(data)
y = np.array(labels)

print("Classes:", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model = SVC(kernel='linear')

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,pred))