# 2d-sdf-net-main
This is the code that combine deepSDF, deepSDF(with gradient constraint), SAL and SALD.

# get start
1. sampler.py (if circle/sector, run generate_circle.py/generate_pie.py instead)
2. (if sal/sald) sald_post_process.py
3. trainer.py (imgs are one tensorboard)
4. (to get result image of high quality)renderer.py
    change img_size and dpi_info(from 100/100 to 1200/1000)

# information
1. models are stored in "new_model/"
2. logs are stored in "new_logs/8_25/" (along with imgs)
3. mySALDnet.py for deepSDF/deepSDF(with gradient), mySALDReal.py for SAL/SALD
    

# bugs
（这部分我用中文写清楚一点）
1. 因为需要对网络输出的梯度再求梯度，SALD和deepSDF（with gradient）用了softplus（SALD里推荐的），但是刚刚想到deepSDF和SAL其实应该用ReLU，不过因为softplus和ReLU整体差不多，只是一个smoothed的ReLU，所以应该不会影响太多。。
2. 算出来的梯度是整体相反的（因为有些时候画shape是顺时针画点，有些时候是逆时针画点，而我计算梯度方向是依赖drawer里画点的顺序，我写计算梯度的时候和做实验用的shape刚好画的方向是反的，当时懒得再动代码，就直接改了loss function的符号），所以deepSDF（with gradient）里的梯度loss是用的加号而不是减号（但是circle和sector我写的是方向对的梯度，所以如果跑circle和sector还要把loss里的加号改成减号）。
