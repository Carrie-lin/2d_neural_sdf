from torch.autograd import grad
import torch

# 参数设定模仿IGR
# inputs：要对什么求偏导，此处为网络输入，（N，3）的点序列
# outputs：被求导，此处为网络输出，（N，1）
# grad_outputs：外部梯度
# create/retain_graph：生成、保留计算图（用于后续bp，而不是单纯输出梯度“值”）
# only_inputs：只保留和inputs相关的grad，其他的梯度不会保留
def myGradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device='cpu').cuda()
    ori_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,  # 对于多维输出，传入与output shape一致的外部梯度
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )  # 输出为tuple，需要取出其中的tensor
    points_grad = ori_grad[0][:, -2:]  # points_grad的shape和网络input一致
    return points_grad



def cosVector(x,y):
    if(len(x)!=len(y)):
        print('error input,x and y is not in the same space')
        return
    result1=0.0
    result2=0.0
    result3=0.0
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    #print(result1)
    #print(result2)
    #print(result3)
    #print("result is ",float(result1/((result2*result3)**0.5))) #结果显示
    #print("x:",x," y:",y)
    return (result1/((result2*result3)**0.5))