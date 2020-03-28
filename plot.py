import matplotlib.pyplot as plt
import numpy as np

x30, y30 = 40.9, 81  #svm
x31, y31 = 62.2, 81
x6, y6 = 6.8, 83.06 #bagging
x10, y10 = 7.72, 92.44 #m2 lite
x11, y11 = 10.04, 92.44 #m2 lite multi
x4, y4 = 3.72, 89.50 #m3 keras
x5, y5 = 0.878, 93.57 #m2 pytorch

x1 = [x10,x11]
y1 = [y10,y11]
x3 = [x30,x31]
y3 = [y30,y31]


fig, ax = plt.subplots()
ax.set_xlabel(r'FPS', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)

ax.scatter(x10,y10,label='MobileNetV2 in Tflite',s=110,c='m')
ax.scatter(x11,y11,s=200,c='m')
plt.plot(x1,y1,c='m')


ax.scatter(x30,y30,label='SVM',c='r',s=50)
ax.scatter(x31,y31,c='r',s=200)
plt.plot(x3,y3,c='r')


ax.scatter(x4,y4,label='MobileNetV3 in Keras',s=200)
ax.scatter(x5,y5,label='MobileNetV2 in Pytorch',s=200)
ax.scatter(x6,y6,label='Bagging',s=200)


ax.scatter(35,70,c='white')
ax.scatter(10,99,c='white')

plt.title('Trade-off for accuracy and FPS',fontsize=25)
plt.legend()
plt.show()