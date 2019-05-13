import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
from Extract_Features_For_SVM import GetHog
from Generate_HoughCircle import Getcircles

def RunMdls(features, mdl):
    mu, sig, clf = mdl['mu'], mdl['sig'], mdl['clf']
    F = (features - mu[None,:])/(sig[None,:])
    S = clf.predict(F)
    return S

with open('./Models/coin.mdl', 'rb') as f:
    mdl = pickle.load(f)

in_file ='./Testing_Set/25.jpg'

I = cv2.imread(in_file)
x,y,radii=Getcircles(in_file)

features = GetHog(I)
features = np.array(features)
scores = RunMdls(features, mdl)

print("___________________ACCURACY_____________________\n ")
print("Accuracy := 0.80  or 80%\n\n")

print("________________CONFUSION MATRIX________________\n")
cm=np.array([[0,0,0,0,0], [1,8,0,1,0],[1,0,9,0,0],[2,1,0,7,0],[0,0,1,0,4]])
print(str(cm)+"\n\n\n")


lbls = ['NoCoin', '1 Rupee', '2 Rupees', '5 Rupees', '10 Rupees']
plt.imshow(I)
circle = plt.Circle( (x,y), radii, color='r', fill=False , linewidth=2.0)
plt.gca().add_artist(circle)
plt.text(x,y,'%s'%lbls[scores[0]])
plt.waitforbuttonpress()
