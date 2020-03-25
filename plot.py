import matplotlib.pyplot as plt
import numpy as np

x3, y3 = 24.35, 81  #svm
x6, y6 = 6.8, 83.06 #bagging
x10, y10 = 7.72, 92.44 #m2 lite
x11, y11 = 10.04, 92.44 #m2 lite multi
x4, y4 = 3.72, 89.50 #m3 keras
x5, y5 = 0.878, 93.57 #m2 pytorch

x1 = [x10,x11]
y1 = [y10,y11]


fig, ax = plt.subplots()
ax.set_xlabel(r'FPS', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)

ax.scatter(x10,y10,label='MobileNetV2 in Tflite',s=110,c='m')
ax.scatter(x11,y11,s=200,c='m')
plt.plot(x1,y1,c='m')


ax.scatter(x3,y3,label='SVM',c='g',s=50)
ax.scatter(x4,y4,label='MobileNetV3 in Keras',s=200)
ax.scatter(x5,y5,label='MobileNetV2 in Pytorch',s=200)
ax.scatter(x6,y6,label='Bagging',s=200)


ax.scatter(35,70,c='white')
ax.scatter(1,99,c='white')

plt.title('Trade-off for accuracy and FPS',fontsize=25)
plt.legend()
plt.show()