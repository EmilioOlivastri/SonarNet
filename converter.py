import os
from pathlib import Path
import cv2

def main() :

    dir_in = Path('../plet/imgs/')
    dir_out = Path('../plet/imgs_gray/')

    list_imgs =  os.listdir(dir_in)

    for img_name in list_imgs :
        img = cv2.imread(str(dir_in) + '/' +  img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(str(dir_out) + '/' +  img_name, gray)

if __name__ == '__main__' :
    main()