import os
from PIL.Image import new
import numpy as np
import cv2
import math
from torch import nn
import torch
from mySALDnet import SALNetwork
from mySALDReal import SALNetworkReal
from shape import Polygon
import json
import matplotlib.pyplot as plt
from commonFunc import *
import json
from torch.utils.tensorboard import SummaryWriter

VISUAL_DATA_PATH="../datasets/visual_data/"
MODEL_PATH = '../new_model/'
DATA_PATH = '../shapes/normalized/'
MASK_PATH = '../shapes/masks/'
TRAINED_PATH = '../results/trained_heatmaps/'
TRUE_PATH = '../results/true_heatmaps/'
TEST_ACCURACY_DATA_PATH='../datasets/testAccuracy/'
LOG_PATH = '../new_logs/8_25/'
ERROR_PATH='../results/error_visual/'
GRAD_DIFF_PATH='../results/grad_diff/'
INTERVAL=80


# Adapted from https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb
def plot_sdf(sdf_func, device, res_path, name, img_name,epoch_num,mask_path,
             img_size=100, is_net=False, show=False,expid="hhh",dpi_info=100):
    writer = SummaryWriter(LOG_PATH+"/"+expid)  

    # parameters
    low = -0.5
    high = 0.5
    grid_size = img_size
    margin = 2e-3
    max_norm = 0.3  # Normalizing distance

    # generate points set as standard input
    grid = np.linspace(low, high, grid_size )
    point_set=[]
    for i in grid:
        for j in grid:
            point_set.append([float(j),float(i)])
    point_set=torch.tensor(point_set,requires_grad=True)
    point_set=point_set.cuda()

    # draw true sdf or draw net learned one
    if not is_net:
        sdf_map = [[sdf_func(np.float_([x_, y_]))
                    for x_ in grid] for y_ in grid]
        sdf_map = np.array(sdf_map, dtype=np.float64)
    else:
        sdf_func.eval()
        sdf_map = sdf_func(point_set)
        sdf_map = sdf_map.detach().cpu().numpy()


    # normalize sdf
    sdf_map=sdf_map.reshape([len(grid),len(grid)])
    max_norm = np.max(np.abs(sdf_map)) if max_norm == 0 else max_norm
    heat_map = sdf_map / max_norm * 157.5 + 127.5
    heat_map = np.minimum(heat_map, 255)
    heat_map = np.maximum(heat_map, 0)



    # Generate a heat map
    store_heat_map=plt.figure(figsize=(10,10),dpi=2000)
    heat_map=np.uint8(heat_map)
    plt.imshow(heat_map, cmap="jet",zorder=0)

    # draw real surface
    with open(f'{MASK_PATH}{name}.json', 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    surface_data=np.array(json_data)
    surface_data=(surface_data * img_size + img_size / 2).astype(float)
    plt.plot(surface_data[:,0], surface_data[:,1], color='w',zorder=10)
    #plt.scatter(surface_data[:,0], surface_data[:,1], c='w',s=0.5,zorder=10)

    # Plot predicted boundary
    low_pos = sdf_map > -margin
    high_pos = sdf_map < margin
    edge_pos = low_pos & high_pos
    new_edge=[]
    for i in range(edge_pos.shape[0]):
        for j in range(edge_pos.shape[1]):
            if edge_pos[i,j]:
                new_edge.append([i,j])
    new_edge=np.array(new_edge)
    if new_edge.shape[0]>0:
        plt.scatter(new_edge[:,1],new_edge[:,0],c='k',s=0.5,alpha=1,zorder=5)

    writer.add_figure(
    'sdf_'+str(epoch_num),
    store_heat_map,
    )
   
    #plt.savefig('../results/test/'+img_name+'.png',dpi=dpi_info)  
    plt.close()




# 可视化error
def visual_error(testXY,visual_sdf_error,visual_grad_error,img_name,expid,epoch_num):
    writer = SummaryWriter(LOG_PATH+"/"+expid)  

    test_point=testXY.detach().cpu().numpy()
    sdf_max=max(visual_sdf_error)
    sdf_min=min(visual_sdf_error)


    # normalize visual sdf error
    for i in range(len(visual_sdf_error)):
        if visual_sdf_error[i]>0:
            visual_sdf_error[i]=visual_sdf_error[i]/sdf_max
        else:
            visual_sdf_error[i]=visual_sdf_error[i]/abs(sdf_min)

    grad_max=max(visual_grad_error)
    for i in range(len(visual_grad_error)):
        visual_grad_error[i]=visual_grad_error[i]/grad_max
    

    # draw gradient error
    grad_error=plt.figure(figsize=(10,10),dpi=500)
    for i in range(len(visual_grad_error)):
        plt.scatter(test_point[i,0],-test_point[i,1],c='r',s=0.5,alpha=visual_grad_error[i])

    # save img

    writer.add_figure(
        'gradError_'+img_name+"_"+str(epoch_num),
        grad_error,
  
        )

    plt.savefig(f'{ERROR_PATH}{expid}_{epoch_num}_grad.png',dpi=500)  
    plt.clf()

    # draw sdf error
    sdf_error=plt.figure(figsize=(10,10),dpi=500)
    for i in range(len(visual_sdf_error)):
        if visual_sdf_error[i]<0:
            plt.scatter(test_point[i,0],-test_point[i,1],c='b',s=0.5,alpha=abs(visual_sdf_error[i]))
        else:
            plt.scatter(test_point[i,0],-test_point[i,1],c='r',s=0.5,alpha=visual_sdf_error[i])
    
    # save img

    writer.add_figure(
        'sdfError_'+str(epoch_num),
        sdf_error,

        )

    plt.savefig(f'{ERROR_PATH}{expid}_{epoch_num}_sdf.png',dpi=500)  
    plt.clf()
    plt.close()


# visualize the gradient difference
def visual_gradient(sdf_func,name,epoch_num,expid):

    writer = SummaryWriter(LOG_PATH+"/"+expid)  

    # 得到表面点坐标及其法向
    testData = []
    testXY=[]
    f = open(f'{VISUAL_DATA_PATH}{name}.txt', 'r')
    line = f.readline()
    while line:
        x, y, sdfx, sdfy = map(lambda n: float(n), line.strip('\n').split(' '))
        testData.append([x, y, sdfx, sdfy])
        testXY.append([x,y])
        line = f.readline()
    f.close()

    point_list=[]
    for i in range(len(testData)):
        point_list.append([testData[i][0],testData[i][1]])
    point_list=np.array(point_list)

    gradient_diff=plt.figure(figsize=(10,10))
    
    # compute real gradient
    testXY=torch.tensor(testXY,requires_grad=True).cuda()
    predSDF=sdf_func(testXY)
    predGrad=myGradient(testXY, predSDF)
    predGrad=predGrad.cpu().tolist()




    for i in range(len(testData)):
        if i%100==0:
            plt.scatter(point_list[i, 0], -point_list[i, 1], s=1,c='r')
            plt.annotate(s="",xytext=(testData[i][0], -testData[i][1]), xy=(testData[i][0]-testData[i][2]*0.05, -testData[i][1]+testData[i][3]*0.05),
                     arrowprops={"facecolor":"r","width": 2, "headlength": 5, "headwidth": 5, "shrink": 0.1})  # 如果arrowprops中有arrowstyle,就不应该有其他的属性，xy代表的是箭头的位置，xytext代表的是箭头文本的位置。
            plt.annotate(s="",xytext=(testData[i][0], -testData[i][1]), xy=(testData[i][0]-predGrad[i][0]*0.05, -testData[i][1]+predGrad[i][1]*0.05),
                     arrowprops={"facecolor":"b","width": 2, "headlength": 5, "headwidth": 5, "shrink": 0.1})  # 如果arrowprops中有arrowstyle,就不应该有其他的属性，xy代表的是箭头的位置，xytext代表的是箭头文本的位置。
            

    # draw real surface
    with open(f'{MASK_PATH}{name}.json', 'r', encoding='utf8')as fp:
        json_data = json.load(fp)

    surface_data=np.array(json_data)
    plt.plot(surface_data[:,0], -surface_data[:,1], color='grey')

    try:
        writer.add_figure(
        'gradientDiff_'+name+"_"+str(epoch_num),
        gradient_diff,
        )

        plt.savefig(f'{GRAD_DIFF_PATH}{expid}_{epoch_num}_sdf.png',dpi=500)  
    
    except:
        print(str(epoch_num)," wrong!")
    
    plt.close()



if __name__ == '__main__':
    mode = ''
    mode="trained"
    while mode != 'trained' and mode != 'true':
        print('Choose mode (trained/true):')
        mode = input()

    print('Enter shape name:')
    #name = input()
    name="myCircle_1_3000"
    net_name="sal myCircle with surface points  sample_3000  seed_1 batchsize_6400 False 08_25__21_13_33_0.001_900"

    if mode == 'trained':
        net = True
        path = TRAINED_PATH
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device}!')

        func = SALNetwork().to(device)
        if os.path.exists(f'{MODEL_PATH}{net_name}.pth'):
            func.load_state_dict(torch.load(f'{MODEL_PATH}{net_name}.pth'))
        else:
            print('Error: No trained data!')
            exit(-1)
    else:
        net = False
        path = TRUE_PATH
        device = 'cpu'
        shape = Polygon()
        shape.load(DATA_PATH, name)
        func = shape.sdf

    print('Plotting results...')
    plot_sdf(func, device, res_path=path, name=name, img_name=net_name,epoch_num=10,mask_path=MASK_PATH, is_net=net, show=False)    
    '''
    # 得到表面点坐标及其法向
    testData=[]
    f = open(f'{TEST_ACCURACY_DATA_PATH}{name}.txt', 'r')
    line = f.readline()
    while line:
        x, y, sdfx, sdfy = map(lambda n: float(n), line.strip('\n').split(' '))
        testData.append([x, y, sdfx, sdfy])
        line = f.readline()
    f.close()

    # 得到x y
    testXY=[]
    testGrad=[]
    for i in range(len(testData)):
        testXY.append([testData[i][0],testData[i][1]])
        testGrad.append([testData[i][2],testData[i][3]])

    #print(len(testData))
    # 得到xy和grad的tensor
    testXY=torch.tensor(testXY,requires_grad=True).to(device)
    testGrad=torch.tensor(testGrad).to(device)

    loss_fn = nn.L1Loss().to(device)

    visual_sdf_error=[]
    visual_grad_error=[]
    sdf_diff_list=[]
    cos_dis_list=[]

    # 得到sdf diff
    test_pred_sdf=func(testXY)
    ideal_sdf=torch.zeros(test_pred_sdf.shape).to(device)
    test_sdf_diff=loss_fn(test_pred_sdf,ideal_sdf)
    test_sdf_rmse=0
    for i in range(test_pred_sdf.shape[0]):
        test_sdf_rmse+=pow((test_pred_sdf[i].cpu()-ideal_sdf[i].cpu()),2)
        visual_sdf_error.append((test_pred_sdf[i].cpu()-ideal_sdf[i].cpu()).detach())
    test_sdf_rmse=test_sdf_rmse/(test_pred_sdf.shape[0])
    test_sdf_rmse=math.sqrt(test_sdf_rmse)
    sdf_diff_list.append(float(test_sdf_diff.tolist()))

    # 得到梯度diff
    test_grad_sdf=myGradient(testXY,test_pred_sdf)
    cos_diff_total=0
    for i in range(test_grad_sdf.shape[0]):
        cos_diff_total+=abs(cosVector(test_grad_sdf[i].cpu().tolist(),testGrad[i].cpu().tolist()))
        visual_grad_error.append(1-abs(cosVector(test_grad_sdf[i].cpu().tolist(),testGrad[i].cpu().tolist())))
    cos_diff_avg=cos_diff_total/test_grad_sdf.shape[0]
    cos_dis_list.append(cos_diff_avg)
    
    
    visual_error(testXY,visual_sdf_error,visual_grad_error,"lgy_test_accuracy")
    '''


    print('Done!')
