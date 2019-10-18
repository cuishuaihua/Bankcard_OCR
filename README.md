# Bankcard_OCR
本代码简易的实现了银行卡识别的功能，通过深度学习（CTPN、Densenet、CTC）实现银行卡号的识别。

## 1、快速开始：
  本代码仅在Ubuntu下通过测试
  #### 1.1环境部署
  ```angular2
  sh setup.sh
  ```
   #### 1.2将测试图片放入demo/test_images目录
   #### 1.3执行demo文件下的demo.py，检测结果会保存到demo/test_result中
  ```angular2
  python demo.py
  ```
      
## 2、效果展示：
![效果图1](https://github.com/taigege/Bankcard_OCR/blob/master/demo/test_result/card_1.jpg)
![效果图2](https://github.com/taigege/Bankcard_OCR/blob/master/demo/test_result/result1.PNG)
    
## 3、训练：
### 3.1不定长文本识别训练
   #### 3.1.1进入data_processing文件下，根据自己需要修改char_std_5990.txt里的内容，然后按照如下顺序依次执行
   ```angular2
  1、data_enhancement.py
  2、train.py
  ```
   #### 3.1.2训练完成的权重文件存放在/data_processing/models下
   #### 3.1.2将训练好的权重替换densenet/models/weights_densenet.h5
### 3.2CTPN训练：
 #### 3.2.1训练代码和银行卡的训练集上传到了百度云：链接：https://pan.baidu.com/s/1AbUwbAs_SutehBCBnp881A&shfl=sharepset 提取码：5r4j 

图片存放在text-detection-ctpn-untagged\data\VOCdevkit2007\VOC2007\JPEGImages下，有2000张左右的训练集。
 #### 3.2.2具体训练过程参考：https://github.com/eragonruan/text-detection-ctpn
      
## 代码来源：
[1] https://github.com/YCG09/chinese_ocr
