import fastbook
import fastai
from fastai.vision.all import *
from fastbook import *

"""
Tfms
"""
def random_rotation(x):
    '''
    Return a float tensor equal in size to batch with random
    rotations multiples of 90. 
    
    x: full batch tensor
    '''
    size = x.size(0)
    result = torch.zeros(([size]), dtype=float, device='cuda')
    
    for i in range(size):
        degree = torch.rand(1)
        if (degree < 0.25):
            result[i] = 0.0
        elif (degree < 0.5):
            result[i] = 90.0
        elif (degree < 0.75):
            result[i] = 180.0
        else:
            result[i] = 270.0
    
    return result

class GaussianNoise(Transform):
    order = 100 #after normalize
    
    def encodes(self, x:TensorImage):
        noise = torch.normal(
            mean=0.0, 
            std=0.005,
            size=x.shape,
            device='cuda'
        )
        return x + noise

"""
Learner
"""
print("========================================")
print("Clasificacion morfologica: Clasificacion")
print("========================================")

import sys
import os

project_root_path = os.path.dirname(sys.argv[0])
project_root_path = project_root_path if project_root_path else "."

learn = load_learner(project_root_path+"/models/classification.pkl", cpu=False)
print("\nModel loaded!")

img = PILImage.create(sys.argv[1])
preds = learn.predict(img)

print(f"\nMorfología predicha: {preds[0]}")
