# deeplabv3plus

## Installations

```
tensorflow>=2.5.2
efficientnet==1.1.1
```
models can be build with Keras or Tensorflow frameworks use keras and tfkeras modules respectively efficientnet.keras / efficientnet.tfkeras at line number 205 in deeplab.py

## Usage
```
from deeplab import Deeplabv3Plus
model =  Deeplabv3Plus(weights = None,input_shape=(512, 512, 3), classes=2, backbone='efficientnetb0',OS=16, activation='softmax')
print(model.summary())  
```
