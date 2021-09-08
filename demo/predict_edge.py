import fastbook
import fastai
from fastai.vision.all import *
from fastbook import *

"""
CSV
"""
#df = pd.read_csv(dataset_dir_path+"benja_ppales_params_low_v4.csv")
#df["majorAxis"] = df["r_edge"]
#df["minorAxis"] = df["majorAxis"] * df["q"]

"""
GETTERS
"""
def get_x(r):
    return png_folder_path + r['gal_id'] + '.png'

def get_y(r):
    return (tensor([r['pa'], r['xcH'], r['ycH'], r['majorAxis'], r['minorAxis']]))

"""
TFMS
"""
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


class RotateTfm(Transform):
    split_idx  =  0 # Only apply to the training set
    order = 101
    def encodes(self, x:Tensor):
        if isinstance(x, TensorImage):
            """
            TensorImages
            """
            # Generate random values between 0 and 3
            # 0: 0.0, 1: 90.0, 2: 180.0, 3: 270.0
            self.rotation = torch.randint(
                low=0, high=4, size=(x.shape[0],), 
                dtype=torch.int, device='cuda'
            )
            
            for img_idx in range(x.shape[0]):
                for ch_idx in range(x.shape[1]):
                    for r in range(self.rotation[img_idx]):
                        x[img_idx][ch_idx] = torch.rot90(
                            x[img_idx][ch_idx], 1, [0, 1]
                        )
        else:
            """
            Regression values
            """
            for img_idx in range(x.shape[0]):
                if self.rotation[img_idx]%2 != 0:
                    x[img_idx][0] = (
                        x[img_idx][0] + 90 
                        if x[img_idx][0] < 0 
                        else x[img_idx][0] - 90
                    )

        return x


class MovementTfm(Transform):
    split_idx  =  0
    order = 102 #after normalize
    
    def __init__(self, min:int, max:int):
        super()

        self.min = min
        self.max = max

    def encodes(self, x:Tensor):
        if isinstance(x, TensorImage):
            """
            TensorImages
            """
            self.movements = torch.randint(
                low=self.min, high=self.max, 
                size=(2, x.shape[0]), 
                dtype=torch.int, device='cuda'
            )
            
            for img_idx in range(x.shape[0]):
                for ch_idx in range(x.shape[1]):
                    y_movement = self.movements[0][img_idx]
                    x_movement = self.movements[1][img_idx]
                    
                    x[img_idx][ch_idx] = torch.roll(
                        x[img_idx][ch_idx], 
                        shifts=(y_movement, x_movement),
                        dims=(0,1)
                    )

                    #Remove pixels that fall out of the tensor
                    if(x_movement >= 0):
                        x[img_idx][ch_idx][:,0:x_movement] = 0
                    else:
                        x[img_idx][ch_idx][:,x_movement:] = 0
                        
                    if(y_movement >= 0):
                        x[img_idx][ch_idx][0:y_movement] = 0
                    else:
                        x[img_idx][ch_idx][y_movement:] = 0
        else:
            """
            Regression values
            """
            for img_idx in range(x.shape[0]):
                x[img_idx][1] += self.movements[0][img_idx] #X
                x[img_idx][2] += self.movements[1][img_idx] #Y
                
        return x

"""
Loss y Metric
"""
def mae_angle(preds, targs):
    if len(preds.shape) != 1:
        preds = torch.transpose(preds, 0, 1)
        targs = torch.transpose(targs, 0, 1)

    errors =  torch.zeros(preds.shape, dtype=torch.float)
    
    pa1_errors = (preds[0]-targs[0]).abs()
    pa2_errors = (preds[0]-(targs[0]+180)).abs()
    pa_errors = torch.min(pa1_errors, pa2_errors)
    
    pa3_errors = (preds[0]-(targs[0]-180)).abs()
    pa_errors = torch.min(pa_errors, pa3_errors)

    errors[0] = pa_errors

    for i in range(1, 5):
        errors[i] = tensor((preds[i]-targs[i]).abs())

    return errors.mean()

def mae_label_smoothing_angle(preds, targs):
    targs += (torch.rand(targs.size(1),device='cuda')-0.5)*0.1

    return mae_angle(preds, targs)


"""
save image
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def save_edge(x, y, img_path):
    if isinstance(x, TensorImage):
        x = x.data
        if x.shape[0]<5: x=x.permute(1,2,0)
        
    img = x
    fig, ax = plt.subplots()
    ax.imshow(img)

    ax.add_patch(
        Ellipse(
            (y[1], y[2]), 
            width=y[4]*2, 
            height=y[3]*2,
            angle=180-y[0],
            edgecolor='red',
            facecolor='none',
            linewidth=2
        )
    )

    values = f"""
PA = {y[0]:.3f}
xcH = {y[1]:.3f}
ycH = {y[2]:.3f}
Major axis = {y[3]:.3f}
Minor axis = {y[4]:.3f}
"""
    ax.set_xlabel(values)
    
    #plt.savefig(img_path)
    
    plt.show()
    
    


"""
Learner
"""
print("==============================")
print("Deteccion de bordes: Regresion")
print("==============================")

import sys
import os

project_root_path = os.path.dirname(sys.argv[0])
project_root_path = project_root_path if project_root_path else "."

learn = load_learner(project_root_path+"/models/edge_detection.pkl", cpu=False)
print("\nModel loaded!")

img_path = sys.argv[1]
img = PILImage.create(sys.argv[1])
preds = learn.predict(img)[1]

result = f"""
Regresion de la elipse del borde de la galaxia:
    Angulo (pa):      		{preds[0]:.4f} 
    Centro en el eje X (xcH):	{preds[1]:.4f} 
    Centro en el eje Y (ycH):	{preds[2]:.4f} 
    Semi-eje mayor (majorAxis):	{preds[3]:.4f} 
    Semi-eje menor (minorAxis):	{preds[4]:.4f}
"""
print(result)

saved_image_path = img_path[:-4]+"_edge.png"
save_edge(img, preds, saved_image_path)
print(f"\nImage saved as '{saved_image_path}'")

