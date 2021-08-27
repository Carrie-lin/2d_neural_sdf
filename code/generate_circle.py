import numpy as np
import matplotlib.pyplot as plt
import cmath
import torch
import random
import json

# 产生圆
radius=0.3
normalize_max=0.5
sample_num=3000
seed_num=1
split_ratio=0.8
n=sample_num
m=sample_num
test_accuracy_data_path='../datasets/testAccuracy/'
TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
mask_path = '../shapes/masks/'
VISUAL_DATA_PATH='../datasets/visual_data/'

save_name='myCircle_'+str(seed_num)+"_"+str(sample_num)
var=(0.0025, 0.00025)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_poly(rad, num_p):
    """
    返回规则num_p边行的各个点
    :param rad: 几何中心到顶点的距离
    :param num_p: 有多少边
    :return: 二维list  N*2
    """
    r = [rad for i in range(num_p)]
    theta = [cmath.pi / num_p * 2 * i for i in range(num_p)]
    ploy = [[cmath.rect(*i).real, cmath.rect(*i).imag] for i in zip(r, theta)]
    ploy=np.array(ploy)
    return ploy

# fix seed
setup_seed(seed_num)


# uniform sampling
r = np.sqrt(np.random.uniform(0, 1, size=(n, 1))) / 2
t = np.random.uniform(0, 2 * np.pi, size=(n, 1))
# Transform to Cartesian coordinate
uniform_points = np.concatenate((r * np.cos(t), r * np.sin(t)), axis=1)

# surface gaussian
circle_surface=get_poly(radius,n)
noise_1 = np.random.normal(loc=0, scale=np.sqrt(var[0]), size=circle_surface.shape)
noise_2 = np.random.normal(loc=0, scale=np.sqrt(var[1]), size=circle_surface.shape)
surface_gaussian=np.concatenate((circle_surface + noise_1, circle_surface + noise_2), axis=0)

# merge samples
all_samples=np.concatenate((surface_gaussian,uniform_points),axis=0)

# compute sdf
sdf_total=[]
for i in range(all_samples.shape[0]):
    sdf_total.append([all_samples[i,0],all_samples[i,1],np.linalg.norm(all_samples[i])-radius,-100])
    #sample_grad.append(all_samples[i])
sdf_total=np.array(sdf_total)


# compute gradient
grad_total=[]
for i in range(circle_surface.shape[0]):
    grad_total.append([circle_surface[i,0],circle_surface[i,1],circle_surface[i,0],circle_surface[i,1]])
grad_total=np.array(grad_total)

# merge sdf and grad
total=np.concatenate((grad_total,sdf_total),axis=0)

# 分装进test和train两个部分
train_size = int(len(total) * split_ratio)
choice = np.random.choice(range(total.shape[0]), size=(train_size,), replace=False)
ind = np.zeros(total.shape[0], dtype=bool)
ind[choice] = True
rest = ~ind
train_data = total[ind]
val_data = total[rest]

# save train data
f = open(f'{TRAIN_DATA_PATH}{save_name}.txt','w')
for datum in total:
    f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
f.close()

# save val data
f = open(f'{VAL_DATA_PATH}{save_name}.txt','w')
for datum in total:
    f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
f.close()

# save mask
# save the points as json file
filename = mask_path+save_name+'.json'
with open(filename, 'w') as file_obj:
    json.dump(circle_surface.tolist(), file_obj)
print(f'Mask json = {filename}')

# get visual data
visual_data=[]
for i in range(total.shape[0]):
    cur_visual_piece = [total[i,0].tolist(),total[i,1].tolist(), total[i,0].tolist(),total[i,1].tolist()]
    visual_data.append(cur_visual_piece)
visual_data=np.array(visual_data)
f = open(f'{VISUAL_DATA_PATH}{save_name}.txt', 'w')
for datum in visual_data:
    f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
f.close()

#print(visual_data)


# get_test_points
test_p=10000
test_points=get_poly(radius,test_p)
test_info=np.concatenate((test_points,test_points),axis=1)
f = open(f'{test_accuracy_data_path}{save_name}.txt','w')
for datum in test_info:
    f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
f.close()





# visualizations
'''
plt.figure(figsize=(10,10))
plt.scatter(surface_gaussian[:,0],surface_gaussian[:,1],c='b',s=1)
plt.scatter(uniform_points[:,0],uniform_points[:,1],c='r',s=1)

plt.show()
'''

