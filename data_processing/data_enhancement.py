import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os


# TOTAL=5000  #生成样本种子数量  注：真实样本数量为 TOTAL * 10
print("注：实际生成样本数为 种子数 * 10 ！！！")
TOTAL=int(input("输入要生成的种子数："))

path_image=r'./images_base/'
list_path=os.listdir(path_image)
loca=["0,0,30,46","30,0,60,46","60,0,90,46","90,0,120,46"]
print(list_path)
ALLIMAGE=len(list_path)#得到给出的训练集数量
print("总共{0}张图片，需要生成{1}张图片\n------------正在生成中------------------".format(ALLIMAGE,TOTAL*10))

def cut_image(imge,l2,l3,l0=0,l1=46):
    src2 = imge[l0:l1, l2:l3]
    return src2

if not os.path.exists("./images"):
    os.mkdir("./images")
# else:


for i in range(TOTAL):
    rad_num=np.random.randint(0,ALLIMAGE,(1,3))
    # [[109 812 758]]
    img_path=[]
    for j in rad_num[0]:
        img_path.append([r"./images_base/"+str(list_path[j])])
    # print(img_path)
    # [['./images/845_c_0.png'], ['./images/0100v_0.png'], ['./images/0024e_0.png']]
    # img_path=img_path[0]
    imge1=cv2.imread(" ".join(img_path[0]))
    imge2 = cv2.imread(" ".join(img_path[1]))
    imge3 = cv2.imread(" ".join(img_path[2]))

    src1=cut_image(imge3, 0, 60)

    # tmp=np.zeros((46,300,3))
    tmp = np.zeros((46, 300, 3), np.uint8)
    tmp[0:46,0:120]=imge1
    tmp[0:46, 120:240]=imge2
    tmp[0:46, 240:300]=src1
    # cv2.imshow("1", tmp)

    name=" ".join(img_path[0]).split("/")[2][0:4]+" ".join(img_path[1]).split("/")[2][0:4]+" ".join(img_path[2]).split("/")[2][0:2]+"I"
    # print("./out_image/"+name)

    tmp=cv2.resize(tmp,(280, 32))   #重述大小
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY) #转灰度

    datagen = ImageDataGenerator(   #数据增强
        rotation_range=2,
        width_shift_range=0.01,
        height_shift_range=0.015,
        zoom_range=0.06,
        fill_mode='nearest')
    x = img_to_array(tmp)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,  # save_to_dir 文件夹   prefix图片名字   format格式
                              save_to_dir='images', save_prefix=name, save_format='jpg'):
        i += 1
        if i > 10:  # 如果不break会无限循环
            break  # otherwise the generator would loop indefinitely

    # cv2.imwrite("./out_data/" + name, tmp)


print("=====描述文件======")
path_image=r'./images/'
list_path=os.listdir(path_image)
f1=open("./data_train.txt",'w')
f2=open("./data_test.txt",'w')

# print(list_path)
for i in list(list_path)[0:int(len(list_path)*0.8)]: #划分80%训练
    f1.write(""+str(i))
    label=list(i)[0:10]
    for j in label:
        if j=="_":
            f1.write(" " + str(11))
        else:
            f1.write(" "+str(int(j)+1))
    f1.write("\n")

for i in list(list_path)[int(len(list_path)*0.8):]:
    f2.write(""+str(i))
    label=list(i)[0:10]
    for j in label:
        if j=="_":
            f2.write(" " + str(11))
        else:
            f2.write(" "+str(int(j)+1))
    f2.write("\n")
print("over")