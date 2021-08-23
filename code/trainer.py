import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import math
import time
from torch.utils.tensorboard import SummaryWriter
#from torch.autograd import grad

from net import SDFNet
from mySALDnet import SALNetwork
from loader import SDFData
from renderer import plot_sdf,visual_error,visual_gradient
from commonFunc import *

TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
TEST_ACCURACY_DATA_PATH='../datasets/testAccuracy/'
MODEL_PATH = '../models/'
RES_PATH = '../results/trained_heatmaps/'
MASK_PATH = '../shapes/masks/'
LOG_PATH = '../new_logs/'

# shape11 
# shape9 +

if __name__ == '__main__':
    
    learning_rate = 1e-4
    epochs = 20000
    #epochs=100
    regularization = 0  # Default: 1e-2
    delta = 0.1  # Truncated distance
    withGrad=False
    SEED_NUM=1
    SAMPLE_NUM=3000
    batch_size=64
    #batch_size = 3*SAMPLE_NUM
    grad_weight=0.1
    withSurfacePoints=False
    if withSurfacePoints:
        ifSur="with surface points "
    else:
        ifSur=""
    ini_name="shape8"
    aaaa=time.strftime("%m_%d__%H_%M_%S", time.localtime())
    expid=ini_name+ifSur+" sample_"+str(SAMPLE_NUM)+"  seed_"+str(SEED_NUM)+" _batchsize"+str(batch_size)+" "+str(withGrad)+" "+aaaa+"_+"
    print(expid)

    name=ini_name+"_"+str(SEED_NUM)+"_"+str(SAMPLE_NUM)

    # train数据 val测试数据
    train_data = SDFData(f'{TRAIN_DATA_PATH}{name}.txt')
    val_data = SDFData(f'{VAL_DATA_PATH}{name}.txt')

    # dataloader
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(f'Using {device}!')

    # 先load model
    model = SALNetwork().to(device)
    #if os.path.exists(f'{MODEL_PATH}{name}.pth'):
    #    model.load_state_dict(torch.load(f'{MODEL_PATH}{name}.pth'))

    # 损失 优化器
    loss_fn = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)


    writer = SummaryWriter(LOG_PATH+"/"+expid)
    total_train_step = 0
    total_val_step = 0

    start_time = time.time()
    cos_dis_list=[]
    sdf_diff_list=[]
    sdf_rmse_list=[]
    cos_rmse_list=[]

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



    for t in range(epochs):
        
        if t==50:
            learning_rate=1.5e-5
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
        #print(f'Epoch {t + 1}\n-------------------------------')
        
        # Training loop
        model.train()
        size = len(train_dataloader.dataset)
        for batch, (xy, sdf) in enumerate(train_dataloader):
            # 优化sdf的数据
            gt_sdf = []
            xy_sdf=[]

            # 优化grad的数据
            gt_grad=[]
            xy_grad=[]

            ifGrad=False
            ifSDF=False
            loss=0

            for k in range(sdf.shape[0]):
                if (sdf[k][1] == -100):
                    gt_sdf.append(sdf[k][0].tolist())
                    xy_sdf.append([xy[k][0].tolist(),xy[k][1].tolist()])
                    ifSDF=True
                else:
                    gt_grad.append(sdf[k].tolist())
                    xy_grad.append([xy[k][0].tolist(),xy[k][1].tolist()])
                    ifGrad=True
            xy_sdf,xy_grad,gt_sdf,gt_grad = torch.tensor(xy_sdf),torch.tensor(xy_grad,requires_grad=True),torch.tensor(gt_sdf),torch.tensor(gt_grad)
            xy_sdf,xy_grad,gt_sdf,gt_grad = xy_sdf.to(device), xy_grad.to(device), gt_sdf.to(device),gt_grad.to(device)

            if ifSDF:
                pred_sdf = model(xy_sdf)
                #print("before",gt_sdf.shape)
                gt_sdf = torch.reshape(gt_sdf, pred_sdf.shape)
                #print("after",gt_sdf.shape)
                loss = loss_fn(torch.clamp(pred_sdf, min=-delta, max=delta), torch.clamp(gt_sdf, min=-delta, max=delta))

            '''
            if ifGrad:
                pred_grad=model(xy_grad)
            
                if withSurfacePoints:
                    grad_gt_sdf=torch.zeros(pred_grad.shape).cuda()
                    loss+=loss_fn(torch.clamp(pred_grad, min=-delta, max=delta), torch.clamp(grad_gt_sdf, min=-delta, max=delta))
            '''
       

            if withGrad and ifGrad:
                pred_grad=model(xy_grad)
                my_grad=myGradient(xy_grad,pred_grad)
                #print(my_grad.shape)

                '''
                g1 = my_grad - gt_grad
                g2 = my_grad + gt_grad

                g1_v = torch.norm(g1, dim=1)
                g2_v = torch.norm(g2, dim=1)

                lossGradAll = torch.stack((g1_v, g2_v))
                lossGradAll = torch.transpose(lossGradAll, 0, 1)
                lossGradAll, _ = torch.min(lossGradAll, dim=1)
                lossGradAll = lossGradAll.reshape(pred_grad.shape[0], 1)
                lossGrad = lossGradAll.mean()
                '''

                g1 = my_grad + gt_grad
                g2 = my_grad + gt_grad

                g1_v = torch.norm(g1, dim=1)
                g2_v = torch.norm(g2, dim=1)

                lossGradAll = torch.stack((g1_v, g2_v))
                lossGradAll = torch.transpose(lossGradAll, 0, 1)
                lossGradAll, _ = torch.min(lossGradAll, dim=1)
                lossGradAll = lossGradAll.reshape(pred_grad.shape[0], 1)
                lossGrad = lossGradAll.mean()
                loss+=grad_weight*lossGrad

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 50 == 0:
                loss, current = loss.item(), batch * len(xy)
                #print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

            total_train_step += 1
            if total_train_step % 200 == 0:
                writer.add_scalar('Training loss', loss, total_train_step)

        # Evaluation loop
        model.eval()
        size = len(val_dataloader.dataset)
        #size = len(val_dataloader.dataset)
        val_loss = 0
        val_grad_loss=0
        val_sdf_loss=0
        ifSDF=False
        ifGrad=False
        grad_loss_num=0
        sdf_loss_num=0

        #with torch.no_grad():
        for xy, sdf in val_dataloader:
            # 优化sdf的数据
            gt_sdf = []
            xy_sdf = []

            # 优化grad的数据
            gt_grad = []
            xy_grad = []

            ifGrad=False
            ifSDF=False
            loss=0

            # 对优化梯度和优化sdf的点做区分，并转为tensor进入gpu
            for k in range(sdf.shape[0]):
                if (sdf[k][1] == -100):
                    gt_sdf.append(sdf[k][0].tolist())
                    xy_sdf.append([xy[k][0].tolist(), xy[k][1].tolist()])
                    ifSDF=True
                else:
                    gt_grad.append(sdf[k].tolist())
                    xy_grad.append([xy[k][0].tolist(), xy[k][1].tolist()])
                    ifGrad=True

            xy_sdf, xy_grad, gt_sdf, gt_grad = torch.tensor(xy_sdf), torch.tensor(xy_grad,requires_grad=True), torch.tensor(
                gt_sdf), torch.tensor(gt_grad)
            xy_sdf, xy_grad, gt_sdf, gt_grad = xy_sdf.to(device), xy_grad.to(device), gt_sdf.to(device), gt_grad.to(
                device)

            if ifSDF:
                pred_sdf = model(xy_sdf)
                gt_sdf = torch.reshape(gt_sdf, pred_sdf.shape)

                loss = loss_fn(torch.clamp(pred_sdf, min=-delta, max=delta), torch.clamp(gt_sdf, min=-delta, max=delta))
                val_sdf_loss+=loss
                sdf_loss_num+=1

            '''
            if ifGrad:
                pred_grad=model(xy_grad)
            
                if withSurfacePoints:
                    grad_gt_sdf=torch.zeros(pred_grad.shape).cuda()
                    loss+=loss_fn(torch.clamp(pred_grad, min=-delta, max=delta), torch.clamp(grad_gt_sdf, min=-delta, max=delta))
            ''' 

            if withGrad and ifGrad:
                pred_grad=model(xy_grad)
                my_grad=myGradient(xy_grad,pred_grad)
                #print(my_grad.shape)

                g1 = my_grad + gt_grad
                g2 = my_grad + gt_grad

                g1_v = torch.norm(g1, dim=1)
                g2_v = torch.norm(g2, dim=1)

                lossGradAll = torch.stack((g1_v, g2_v))
                lossGradAll = torch.transpose(lossGradAll, 0, 1)
                lossGradAll, _ = torch.min(lossGradAll, dim=1)
                lossGradAll = lossGradAll.reshape(pred_grad.shape[0], 1)
                lossGrad = lossGradAll.mean()

                val_grad_loss+=lossGrad
                loss+=grad_weight*lossGrad
                grad_loss_num+=1

            val_loss += loss

        val_loss /= size

        if withGrad:
            if ifGrad:
                val_grad_loss/=grad_loss_num
                writer.add_scalar('Val gradient loss', val_grad_loss, total_val_step)
            if ifSDF:
                val_sdf_loss/=sdf_loss_num
                writer.add_scalar('Val sdf loss', val_sdf_loss, total_val_step)


        end_time = time.time()
        #print(f'Test Error: \n Avg loss: {val_loss:>8f} \n Time: {(end_time - start_time):>12f} \n ')

        total_val_step += 1
        writer.add_scalar('Val loss', val_loss, total_val_step)
        
        
        

        # =============================测试效果===========================================================================

        visual_sdf_error=[]
        visual_grad_error=[]

        # 得到sdf diff
        test_pred_sdf=model(testXY)
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

        # =============================测试效果===========================================================================

        writer.add_scalar('cosine distance', 1-cos_diff_avg, total_val_step)
        writer.add_scalar('shape difference', test_sdf_diff, total_val_step)
        writer.add_scalar('shape rmse difference', test_sdf_rmse, total_val_step)

        # 绘制图片
        if t%300==0:
        #if 1:
            if t==0:
                a=1
            else:
                plot_sdf(model, device, res_path=RES_PATH, name=ini_name+"_"+str(SEED_NUM)+"_"+str(SAMPLE_NUM), img_name=ifSur+name+"_"+str(withGrad),epoch_num=t,mask_path=MASK_PATH, is_net=True, show=False,expid=expid)
                #visual_error(testXY,visual_sdf_error,visual_grad_error,img_name=ifSur+name+"_"+str(withGrad),expid=expid,epoch_num=t)
                #visual_gradient(sdf_func=model,name=name,epoch_num=t,expid=expid)
                #visual_error(testXY,visual_sdf_error,visual_grad_error)
        #testShape(name)

    '''
    # 输出cos distance
    stable_cos_dis=cos_dis_list[-200:]
    avg_cos_dis=sum(stable_cos_dis)/len(stable_cos_dis)
    print("Average cosine distance:",1-avg_cos_dis)

    # 输出shape distance
    stable_sdf_diff=sdf_diff_list[-200:]
    avg_sdf_diff=sum(stable_sdf_diff)/len(stable_sdf_diff)
    print("Average sdf distance:",avg_sdf_diff)

    '''
    torch.save(model.state_dict(), f'{MODEL_PATH}{name}.pth')
    print(f'Complete training with {epochs} epochs!')

    #writer.close()




    # Plot results
    print('Plotting results...')
    
    plot_sdf(model, device, res_path=RES_PATH, name=ini_name, img_name=ifSur+name+"_"+str(withGrad),epoch_num=t,mask_path=MASK_PATH, is_net=True, show=False,expid=expid)
    #visual_error(testXY,visual_sdf_error,visual_grad_error,img_name=ifSur+name+"_"+str(withGrad),expid=expid,epoch_num=t)
    #visual_gradient(sdf_func=model,name=name,epoch_num=t,expid=expid)
    print('Done!')

    writer.close()
