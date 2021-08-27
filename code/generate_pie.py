import numpy as np
import matplotlib.pyplot as plt
import cmath
import torch
import math
import random
import json

# 产生圆
radius=0.4
normalize_max=0.5
sample_num=6000
seed_num=1
split_ratio=0.8
n=sample_num
m=sample_num
test_accuracy_data_path='../datasets/testAccuracy/'
TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
mask_path = '../shapes/masks/'
VISUAL_DATA_PATH='../datasets/visual_data/'

save_name='myPie_'+str(seed_num)+"_"+str(sample_num)
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

# 计算两点之间线段的距离
def __line_magnitude(x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude


def __point_to_line_distance(point, line):
    px, py = point
    x1, y1, x2, y2 = line
    line_magnitude = __line_magnitude(x1, y1, x2, y2)
    if line_magnitude < 0.00000001:
        return 9999
    else:
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (line_magnitude * line_magnitude)
        if (u < 0.0000000001) or (u > 1):
            # 点到直线的投影不在线段内, 计算点到两个端点距离的最小值即为"点到线段最小距离"
            ix = __line_magnitude(px, py, x1, y1)
            iy = __line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance = iy
            else:
                distance = ix
        else:
            # 投影点在线段内部, 计算方式同点到直线距离, u 为投影点距离x1在x1x2上的比例, 以此计算出投影点的坐标
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = __line_magnitude(px, py, ix, iy)
        return distance

# input 为np array
def compute_sdf(p):
    #p=p.tolist()
    p1=np.array([0,radius])
    p2=np.array([radius,0])
    diss=0
    dir=[]
    if p[0]>=0 and p[1]>=0:
        dis_circle = np.linalg.norm(p) - radius
        if dis_circle < 0:
            dis_line1 = __point_to_line_distance(p.tolist(), [0,0,p1[0],p1[1]])
            dis_line2 = __point_to_line_distance(p.tolist(), [0, 0, p2[0], p2[1]])
            list1=[abs(dis_circle),dis_line1,dis_line2]
            diss=min(list1)
            if list1.index(diss)==0:
                dir=p
            elif list1.index(diss)==1:
                dir=np.array([-p[0],0])
            else:
                dir = np.array([0, -p[1]])
            diss=-min(list1)
        else:
            diss=dis_circle
            dir = p
    elif p[0]<=0 and p[1]>=radius:
        diss=np.linalg.norm(p-p1)
        dir=p-p1
    elif p[1]<=0 and p[0]>=radius:
        diss = np.linalg.norm(p - p2)
        dir = p - p2
    elif p[0]<=0 and p[1]<=0:
        diss = np.linalg.norm(p)
        dir = p
    elif p[0]<=0 and p[1]>=0 and p[1]<=radius:
        diss = abs(p[0])
        dir=np.array([p[0],0])
    elif p[1] <= 0 and p[0] >= 0 and p[0] <= radius:
        diss = abs(p[1])
        dir = np.array([0, p[1]])
    else:
        print(p)
    return {
        "grad":dir,
        "sdf":diss
    }

#heatmap=np.zeros(100,100)
'''
p_list=[]
res_list=[]
for i in range(-50,50):
    for j in range(-50,50):
        p_list.append([i/50,j/50])
for i in range(len(p_list)):
    res_list.append(compute_sdf(np.array(p_list[i])+0.2)["sdf"])
res_list=np.array(res_list)
res_list=np.reshape(res_list,[100,100])


plt.imshow(res_list, cmap="jet")
plt.colorbar()
#plt.scatter(circle_surface[:,0],circle_surface[:,1],c='g',s=1)
plt.show()


'''



# fix seed
setup_seed(seed_num)


# uniform sampling
r = np.sqrt(np.random.uniform(0, 1, size=(n, 1))) / 2
t = np.random.uniform(0, 2 * np.pi, size=(n, 1))
# Transform to Cartesian coordinate
uniform_points = np.concatenate((r * np.cos(t), r * np.sin(t)), axis=1)


# surface gaussian
circle_surface=get_poly(radius,n)
new_circle_surface=[]
for i in range(circle_surface.shape[0]):
    if circle_surface[i,0]>=0 and circle_surface[i,1]>=0:
        new_circle_surface.append(circle_surface[i])
line=np.linspace(0,radius,int(n/6))
for i in range(int(n/6)):
    new_circle_surface.append([0,line[i]])
    new_circle_surface.append([line[i],0])
circle_surface=np.array(new_circle_surface)
circle_surface=circle_surface-0.2
noise_1 = np.random.normal(loc=0, scale=np.sqrt(var[0]), size=circle_surface.shape)
noise_2 = np.random.normal(loc=0, scale=np.sqrt(var[1]), size=circle_surface.shape)
surface_gaussian=np.concatenate((circle_surface + noise_1, circle_surface + noise_2), axis=0)


# merge samples
all_samples=np.concatenate((surface_gaussian,uniform_points),axis=0)

# compute sdf
sdf_total=[]
for i in range(all_samples.shape[0]):
    sdf_value=compute_sdf(all_samples[i]+0.2)["sdf"]
    sdf_total.append([all_samples[i,0],all_samples[i,1],sdf_value,-100])
sdf_total=np.array(sdf_total)


# compute gradient
grad_total=[]
for i in range(circle_surface.shape[0]):
    sdf_grad=compute_sdf(all_samples[i]+0.2)["grad"]
    grad_total.append([circle_surface[i,0],circle_surface[i,1],sdf_grad[0],sdf_grad[1]])
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
    sdf_grad=compute_sdf(np.array([total[i,0],total[i,1]]))["grad"]
    cur_visual_piece = [total[i,0].tolist(),total[i,1].tolist(), sdf_grad[0],sdf_grad[1]]
    visual_data.append(cur_visual_piece)
visual_data=np.array(visual_data)
f = open(f'{VISUAL_DATA_PATH}{save_name}.txt', 'w')
for datum in visual_data:
    f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
f.close()

#print(visual_data)

'''
# get_test_points
test_p=10000
test_points=get_poly(radius,test_p)
test_info=np.concatenate((test_points,test_points),axis=1)
f = open(f'{test_accuracy_data_path}{save_name}.txt','w')
for datum in test_info:
    f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
f.close()



# visualizations
plt.figure(figsize=(10,10))

plt.scatter(circle_surface[:,0],circle_surface[:,1],c='g',s=1)
plt.scatter(surface_gaussian[:,0],surface_gaussian[:,1],c='b',s=1)
plt.scatter(uniform_points[:,0],uniform_points[:,1],c='r',s=1)

plt.show()



'''