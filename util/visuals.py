import imageio
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter



def display_img(filename, epoch,create_rate,imges,phase,subset_count):

    if phase == 'train':
        pix_img = create_rate.train_result_img
    else:
        pix_img = create_rate.val_result_img

    img = Image.fromarray((imges*255).astype(np.uint8))
    result_img = Image.alpha_composite(img.convert('RGBA'), pix_img)
    result_img = cv2.cvtColor(np.asarray(result_img), cv2.COLOR_RGBA2BGRA)
    # plt.imshow(result_img)
    # plt.show()
    cv2.imwrite(filename, result_img)

    return True
