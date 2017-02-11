import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
	pil_img = Image.fromarray(np.uint8(img))#convert numpy data to PIL data
	pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[3]
label = t_train[3]
print(label)

img = img.reshape(28,28)

img_show(img)
