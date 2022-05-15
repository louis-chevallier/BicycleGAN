import glob, os
from utillc import *
from pathlib import Path
import random
import tqdm
import PIL
import numpy as np
import hed
import torch, cv2
import matplotlib.pyplot as plt

pth = "/mnt/hd2/data/downloads/data0/lsun"
jpg_files = list(Path(pth).glob('*/*/*/*/*.jpg'))
random.shuffle(jpg_files)
EKOX(jpg_files[0:12])
#files = glob.glob(pth, *.jpg, recursive=True)
EKOX(len(jpg_files))
hh, ww = 320, 480
good_size = (hh, ww)
EKOX(hh/ww)

for f in tqdm.tqdm(jpg_files, total=len(jpg_files)) :
    black = np.zeros((hh, ww, 3),dtype=np.uint8)
    stem = f.stem
    im = PIL.Image.open(f)
    im = np.asarray(im)
    #EKOI(im, sz=200)
    #EKOX(im.shape)
    img = resizeImage(im, ht = hh, wt=ww, pad=False)    
    tenInput = torch.FloatTensor(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    #EKOI(img, sz=200)
    tenOutput = hed.estimate(tenInput)
    npres = (tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(np.uint8)
    #EKOI(npres, sz=200)
    
    ret,thresh1 = cv2.threshold(npres,100,255,cv2.THRESH_BINARY)
    #plt.imshow(thresh1); plt.show()    
    #EKOI(thresh1)
    npres = thresh1
    
    kernel = np.ones((2, 2), np.uint8)  
    # Using cv2.erode() method
    #plt.imshow(npres); plt.show()    
    npres = cv2.erode(npres, kernel,iterations=2) 
    #EKOI(npres)
    #plt.imshow(npres); plt.show()
    npres = np.stack([npres]*3, axis=2)
    #EKOX(npres.shape)
    #EKOX(TYPE(npres))
    #EKOX(TYPE(img))
    npres = np.hstack((npres, img))
    
    res = PIL.Image.fromarray(npres)
    res.save("out.png")
    
    #EKO()
    res.save(os.path.join("/media/louis/hyperX/data/lsun", stem + ".png"))
    #break
