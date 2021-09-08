import fastbook
import fastai
from fastai.vision.all import *
from fastbook import *

"""
Getters
"""
import os

def get_y(source):
    '''
    Pre: file = source/morphology_id_field.png
    '''
    path = str(source)
    filename = os.path.basename(path)
    return galaxies_fs.get(filename) # Search in the dictionary

"""
Loss
"""
def rmse_label_smoothing(preds, targs):
    targs += (torch.rand(targs.size(1),device='cuda')-0.5)*0.1
    return ((preds-targs)**2).mean().sqrt()  # rmse

def morph_accuracy(preds, targs):
    morphs_preds = []
    success = 0

    for index, fs in enumerate([preds,targs]):
        for f_index, f in enumerate(fs):
            f_sph = f[0]
            f_disk = f[1]
            f_irr = f[2]

            if index == 0:
                morphs_preds.append(get_morphology_Huertas15(f_sph, f_disk, f_irr))
            else:
                if morphs_preds[f_index] == get_morphology_Huertas15(f_sph, f_disk, f_irr):
                    success += 1
        
    return success/len(preds)

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
print("====================================")
print("Clasificacion morfologica: Regresion")
print("====================================")

import sys
import os

project_root_path = os.path.dirname(sys.argv[0])
project_root_path = project_root_path if project_root_path else "."

learn = load_learner(project_root_path+"/models/regression.pkl", cpu=False)
print("\nModel loaded!")

img = PILImage.create(sys.argv[1])
preds = learn.predict(img)
#print(preds)

result = f"""
Fracciones predichas:
    Esferoide (f_sph):      {preds[0][0]:.4f} 
    Disco (f_disk):         {preds[0][1]:.4f} 
    Irregularidad (f_irr):  {preds[0][2]:.4f} 
    Fuente puntual (f_ps):  {preds[0][3]:.4f} 
    Inclasificable (f_unc): {preds[0][4]:.4f}
"""
print(result)
