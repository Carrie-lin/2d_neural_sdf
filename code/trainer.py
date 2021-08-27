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
from mySALDReal import SALNetworkReal
from loader import SDFData
from renderer import plot_sdf,visual_error,visual_gradient
from commonFunc import *

TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
TEST_ACCURACY_DATA_PATH='../datasets/testAccuracy/'
MODEL_PATH = '../new_model/'
RES_PATH = '../results/trained_heatmaps/'
MASK_PATH = '../shapes/masks/'
LOG_PATH = '../new_logs/8_25/'
SALD_TRAIN_PATH='../datasets/sald/train/'
SALD_VAL_PATH='../datasets/sald/val/'

# test shape
# shape8 
# shape9 
# shape3

if __name__ == '__main__':
    
    learning_rate = 1e-3
    epochs = 200000
    regularization = 0  
    delta = 0.1  # Truncated distance
    withGrad=True # if with gradient constraint
    withSALD=True # if is SAL or SALD (different training data source)
    withSurfacePoints=True # if add |SDFsurface points-0| in loss 
    SEED_NUM=1 
    SAMPLE_NUM=3000 # means 3000*4=12000
    batch_size=6400
    grad_weight=0.1 #the weight of gradient loss in total loss 
    ini_name="shape3" # shape name

    # expid for tensorboard
    ifSur=""
    if withSurfacePoints:
        ifSur=" with surface points "    
    ifSALD=""
    if withSALD:
        ifSALD="sal "
    
    time_info=time.strftime("%m_%d__%H_%M_%S", time.localtime())
    expid=ifSALD+ini_name+ifSur+" sample_"+str(SAMPLE_NUM)+"  seed_"+str(SEED_NUM)+" batchsize_"+str(batch_size)+" "+str(withGrad)+" "+time_info+"_"+str(learning_rate)
    print(expid)

    name=ini_name+"_"+str(SEED_NUM)+"_"+str(SAMPLE_NUM)

    # load train and test data
    if withSALD:
        train_data = SDFData(f'{SALD_TRAIN_PATH}{name}.txt')
        val_data = SDFData(f'{SALD_VAL_PATH}{name}.txt')
    else:
        train_data = SDFData(f'{TRAIN_DATA_PATH}{name}.txt')
        val_data = SDFData(f'{VAL_DATA_PATH}{name}.txt')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    if withSALD:
        model=SALNetworkReal().to(device)
    else:
        model = SALNetwork().to(device)

    #if os.path.exists(f'{MODEL_PATH}{name}.pth'):
    #    model.load_state_dict(torch.load(f'{MODEL_PATH}{name}.pth'))

    loss_fn = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)


    writer = SummaryWriter(LOG_PATH+"/"+expid)
    total_train_step = 0
    total_val_step = 0

    
    cos_dis_list=[]
    sdf_diff_list=[]
    sdf_rmse_list=[]
    cos_rmse_list=[]

    # get data for testing accuracy
    testData=[]
    f = open(f'{TEST_ACCURACY_DATA_PATH}{name}.txt', 'r')
    line = f.readline()
    while line:
        x, y, sdfx, sdfy = map(lambda n: float(n), line.strip('\n').split(' '))
        testData.append([x, y, sdfx, sdfy])
        line = f.readline()
    f.close()

    testXY=[]
    testGrad=[]
    for i in range(len(testData)):
        testXY.append([testData[i][0],testData[i][1]])
        testGrad.append([testData[i][2],testData[i][3]])

    testXY=torch.tensor(testXY,requires_grad=True).to(device)
    testGrad=torch.tensor(testGrad).to(device)



    for t in range(epochs):
        
        # change learning rate 
        if t==100:
            learning_rate=0.7e-3
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
        

        # Training loop
        model.train()
        size = len(train_dataloader.dataset)
        for batch, (xy, sdf) in enumerate(train_dataloader):
            # data for sdf optimization
            gt_sdf = []
            xy_sdf=[]

            # data for gradient optimization
            gt_grad=[]
            xy_grad=[]

            ifGrad=False
            ifSDF=False
            loss=0

            # sort data (for sdf or for gradient optimization)
            for k in range(sdf.shape[0]):
                # if the data is for sdf optimization, we set the final data to be -100 
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
            
            #start_time = time.time()

            # if exitst data for sdf optimization
            if ifSDF:
                
                pred_sdf = model(xy_sdf)
                gt_sdf = torch.reshape(gt_sdf, pred_sdf.shape)
                # sald/sal loss=mean(||f(x)|-gtSDF|)
                # deepSDF/deepSDF(with gradient) loss=mean(|f(x)-gtSDF|)
                if withSALD:
                    lossDistAll = torch.abs(pred_sdf.abs() - gt_sdf)
                    loss = lossDistAll.mean()
                else:
                    loss = loss_fn(torch.clamp(pred_sdf, min=-delta, max=delta), torch.clamp(gt_sdf, min=-delta, max=delta))

            # if exitst data for gradient optimization
            if ifGrad:
                pred_grad=model(xy_grad)
                # the sdf for surface points should be zero
                if withSurfacePoints:
                    grad_gt_sdf=torch.zeros(pred_grad.shape).cuda()
                    loss+=loss_fn(torch.clamp(pred_grad, min=-delta, max=delta), torch.clamp(grad_gt_sdf, min=-delta, max=delta))
                  
            # if we want to compute gradient
            if withGrad and ifGrad:
                # get gradient 
                my_grad=myGradient(xy_grad,pred_grad)
                
                # SALD loss=min(|a+b|,|a-b|)
                # deepSDF(with gradient) loss=|a-b|(here we use a+b but it means a-b since I give the inversed gt gradient)
                if withSALD:
                    g1 = my_grad - gt_grad
                    g2 = my_grad + gt_grad
                else:
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
            #end_time = time.time()  
            '''
            if batch % 50 == 0:
                loss, current = loss.item(), batch * len(xy)
                #print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
            '''
            total_train_step += 1
            if total_train_step % 200 == 0:
                writer.add_scalar('Training loss', loss, total_train_step)

        # Evaluation loop
        # same as training one
        model.eval()
        size = len(val_dataloader.dataset)
        val_loss = 0
        val_grad_loss=0
        val_sdf_loss=0
        ifSDF=False
        ifGrad=False
        grad_loss_num=0
        sdf_loss_num=0

        for xy, sdf in val_dataloader:

            gt_sdf = []
            xy_sdf = []

            gt_grad = []
            xy_grad = []

            ifGrad=False
            ifSDF=False
            loss=0

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
                if withSALD:
                    lossDistAll = torch.abs(pred_sdf.abs() - gt_sdf)
                    loss = lossDistAll.mean()
                else:
                    loss = loss_fn(torch.clamp(pred_sdf, min=-delta, max=delta), torch.clamp(gt_sdf, min=-delta, max=delta))
                val_sdf_loss+=loss
                sdf_loss_num+=1

            
            if ifGrad:
                pred_grad=model(xy_grad)
            
                if withSurfacePoints:
                    grad_gt_sdf=torch.zeros(pred_grad.shape).cuda()
                    loss+=loss_fn(torch.clamp(pred_grad, min=-delta, max=delta), torch.clamp(grad_gt_sdf, min=-delta, max=delta))
            

            if withGrad and ifGrad:

                my_grad=myGradient(xy_grad,pred_grad)

                if withSALD:
                    g1 = my_grad + gt_grad
                    g2 = my_grad - gt_grad
                else:
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

        total_val_step += 1
        writer.add_scalar('Val loss', val_loss, total_val_step)
        
        
        

        # =============================test accuracy===========================================================================
        
        visual_sdf_error=[]
        visual_grad_error=[]

        # get sdf diff
        test_pred_sdf=model(testXY)
        ideal_sdf=torch.zeros(test_pred_sdf.shape).to(device)
        test_sdf_diff=loss_fn(test_pred_sdf,ideal_sdf)
        test_sdf_rmse=0
        for i in range(test_pred_sdf.shape[0]):
            test_sdf_rmse+=pow((test_pred_sdf[i].cpu()-ideal_sdf[i].cpu()),2)
            visual_sdf_error.append((test_pred_sdf[i].cpu()-ideal_sdf[i].cpu()).detach())
        test_sdf_rmse=test_sdf_rmse/(test_pred_sdf.shape[0])
        test_sdf_rmse=math.sqrt(test_sdf_rmse)
        #sdf_diff_list.append(float(test_sdf_diff.tolist()))

        # get gradeint diff
        test_grad_sdf=myGradient(testXY,test_pred_sdf)
        cos_diff_total=0
        for i in range(test_grad_sdf.shape[0]):
            # 这里用了abs是因为写的时候没有考虑到SALD会出现梯度翻转问题，默认大部分形状都是基本吻合，只有方向上小于90度的偏差。
            # 这么写主要是因为当时懒得检查梯度的正负
            cos_diff_total+=abs(cosVector(test_grad_sdf[i].cpu().tolist(),testGrad[i].cpu().tolist()))
            visual_grad_error.append(1-abs(cosVector(test_grad_sdf[i].cpu().tolist(),testGrad[i].cpu().tolist())))
        cos_diff_avg=cos_diff_total/test_grad_sdf.shape[0]
        #cos_dis_list.append(cos_diff_avg)

                
        writer.add_scalar('cosine distance', 1-cos_diff_avg, total_val_step)
        writer.add_scalar('shape difference', test_sdf_diff, total_val_step)
        writer.add_scalar('shape rmse difference', test_sdf_rmse, total_val_step)
        print("epoch:",t," || shape rmse:",test_sdf_rmse," || cos dis:",1-cos_diff_avg)

        # =============================test accuracy===========================================================================

        # draw pic
        if t%300==0:
            if t==0:
                pass
            else:
                plot_sdf(model, device, res_path=RES_PATH, name=ini_name+"_"+str(SEED_NUM)+"_"+str(SAMPLE_NUM), img_name=ifSur+name+"_"+str(withGrad),epoch_num=t,mask_path=MASK_PATH, is_net=True, show=False,expid=expid)
                torch.save(model.state_dict(), MODEL_PATH+expid+"_epoch "+str(t)+'.pth')
       
    print(f'Complete training with {epochs} epochs!')

    print('Done!')

    writer.close()
