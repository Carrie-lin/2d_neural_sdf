import numpy as np
import cv2
from renderer import plot_sdf
from shape import Polygon
import torch
import random
import json

SHAPE_PATH = '../shapes/raw/'
NORM_PATH = '../shapes/normalized/'
NORM_IMAGE_PATH = '../shapes/normalized_images/'
MASK_PATH = '../shapes/masks/'
TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
TEST_ACCURACY_DATA_PATH='../datasets/testAccuracy/'
SAMPLED_IMAGE_PATH = '../datasets/sampled_images/'
HEATMAP_PATH = '../results/true_heatmaps/'
VISUAL_DATA_PATH='../datasets/visual_data/'

CANVAS_SIZE = np.array([800, 800])  # Keep two dimensions the same
SHAPE_COLOR = (255, 255, 255)
POINT_COLOR = (128, 255, 128)

SEED_NUM=1
SAMPLE_NUM=3000
SHAPE_NAME="shape3"
plot_here=False


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ShapeSampler(object):
    def __init__(self, shape_name, shape_path, train_data_path, val_data_path, test_accuracy_data_path,sampled_image_path,
                 norm_path, norm_image_path, mask_path, visual_data_path,split_ratio=0.8, show_image=False):
        """
        :param split_ratio: train / (train + val)
        :param show_image: Launch a windows showing sampled image
        """

        self.shape_name = shape_name
        self.shape_path = shape_path

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_accuracy_data_path=test_accuracy_data_path
        self.sampled_image_path = sampled_image_path
        self.visual_data_path=visual_data_path

        self.norm_path = norm_path
        self.norm_image_path = norm_image_path
        self.mask_path = mask_path

        self.shape = Polygon()
        self.sampled_data = np.array([])
        self.train_data = np.array([])
        self.val_data = np.array([])
        self.test_accuracy_data=np.array([])
        self.visual_data=[]

        self.split_ratio = split_ratio
        self.show_image = show_image

    def run(self):
        self.load()
        self.normalize()
        self.sample()
        self.save()

    def load(self):
        self.shape.load(self.shape_path, self.shape_name)

    def normalize(self, scope=0.4):
        """
        :param scope: Should be less than 0.5
        """
        if self.shape.num == 0:
            return
        # Calculate the center of gravity
        # and translate the shape
        g = np.mean(self.shape.v, axis=0)
        trans_shape = self.shape.v - g
        # Calculate the farthest away point
        # and scale the shape so that it is bounded by the unit circle
        max_dist = np.max(np.linalg.norm(trans_shape, axis=1))
        scaling_factor = scope / max_dist
        trans_shape *= scaling_factor
        self.shape.v = trans_shape

        # Save normalized data
        save_name = self.shape_name

        f = open(f'{self.norm_path}{save_name}.txt', 'w')
        for datum in self.shape.v:
            f.write(f'{datum[0]} {datum[1]}\n')
        f.close()
        print(f'Normalized data path = {self.norm_path}{save_name}.txt')

        ini_scaled_v=self.shape.v
        #print(type(ini_scaled_v))
        scaled_v = np.around(self.shape.v * CANVAS_SIZE + CANVAS_SIZE / 2).astype(int)
        #print(type(scaled_v))
        norm = np.zeros(CANVAS_SIZE, np.uint8)
        cv2.fillPoly(norm, scaled_v[np.newaxis, :, :], SHAPE_COLOR)
        cv2.imwrite(f'{self.norm_image_path}{save_name}.png', norm)
        print(f'Normalized image path = {self.norm_image_path}{save_name}.png')
        #mask = np.zeros(CANVAS_SIZE, np.uint8)
        #mask = np.zeros(CANVAS_SIZE, np.float64)

        # save the points as json file
        filename = self.mask_path+save_name+'_'+str(SEED_NUM)+"_"+str(SAMPLE_NUM)+'.json'
        with open(filename, 'w') as file_obj:
            json.dump(ini_scaled_v.tolist(), file_obj)
        print(f'Mask json = {filename}')

        #cv2.polylines(mask, ini_scaled_v[np.newaxis, :, :], True, SHAPE_COLOR, 1)
        #cv2.imwrite(f'{self.mask_path}{save_name}.png', mask)

        # Plot_sdf
        if plot_here:
            plot_sdf(self.shape.sdf, 'cpu', res_path=HEATMAP_PATH, name=self.shape_name,img_name=self.shape_name+"_"+str(SEED_NUM)+"_"+str(SAMPLE_NUM), epoch_num="ini",mask_path=NORM_IMAGE_PATH,
                 is_net=False, show=False)

    def sample(self, m=SAMPLE_NUM, n=SAMPLE_NUM,tnum=10000, var=(0.0025, 0.00025)):
        """
        :param m: number of points sampled on the boundary
                  each boundary point generates 2 samples
        :param n: number of points sampled uniformly in the unit circle
        :param var: two Gaussian variances used to transform boundary points
        :param tnum: number of points sampled on the boundary for testing the accuracy of shape and gradient
        """

        if self.shape.num == 0:
            return

        # Do uniform sampling
        # Use polar coordinate
        r = np.sqrt(np.random.uniform(0, 1, size=(n, 1))) / 2
        t = np.random.uniform(0, 2 * np.pi, size=(n, 1))
        # Transform to Cartesian coordinate
        uniform_points = np.concatenate((r * np.cos(t), r * np.sin(t)), axis=1)

        # Do Gaussian sampling
        # Distribute points to each edge weighted by length
        total_length = 0
        edge_length = np.zeros(self.shape.num, dtype=np.double)
        for i in range(self.shape.num):
            length = np.linalg.norm(self.shape.v[(i + 1) % self.shape.num] - self.shape.v[i])
            edge_length[i] = length
            total_length += length
        edge_portion = edge_length / total_length
        edge_portion *= m
        edge_num = np.around(edge_portion).astype(int)

        # Do sampling on edges
        direction = (self.shape.v[1] - self.shape.v[0])
        d = np.random.uniform(1e-7, 1, size=(edge_num[0], 1))

        boundary_points = self.shape.v[0] + d * direction
        normal_direc=np.array([[-direction[1],direction[0]]])
        normal_direc=normal_direc/np.linalg.norm(normal_direc)

        for i in range(boundary_points.shape[0]-1):
            cur_normal_direc=np.array([[direction[1],-direction[0]]])
            cur_normal_direc=cur_normal_direc/np.linalg.norm(cur_normal_direc)
            normal_direc=np.concatenate((normal_direc,cur_normal_direc),axis=0)

        grad_data=boundary_points.copy()
        grad_data=np.concatenate((grad_data,normal_direc),axis=1)

        for i in range(1, self.shape.num):
            direction = (self.shape.v[(i + 1) % self.shape.num] - self.shape.v[i])

            d = np.random.uniform(1e-7, 1, size=(edge_num[i], 1))

            cur_points=self.shape.v[i] + d * direction
            if not (cur_points.shape[0] == 0):
                boundary_points = np.concatenate((boundary_points, self.shape.v[i] + d * direction), axis=0)

                # get normal information for surface points
                normal_direc = np.array([[direction[1], -direction[0]]])
                normal_direc = normal_direc / np.linalg.norm(normal_direc)

                for j in range(cur_points.shape[0] - 1):
                    cur_normal_direc = np.array([[direction[1], -direction[0]]])
                    cur_normal_direc = cur_normal_direc / np.linalg.norm(cur_normal_direc)
                    normal_direc = np.concatenate((normal_direc, cur_normal_direc), axis=0)

                # get gradient gt
                cur_grad_data = cur_points.copy()

                cur_grad_data = np.concatenate((cur_grad_data, normal_direc), axis=1)
                grad_data=np.concatenate((cur_grad_data,grad_data),axis=0)

        # Perturbing boundary points
        noise_1 = np.random.normal(loc=0, scale=np.sqrt(var[0]), size=boundary_points.shape)
        noise_2 = np.random.normal(loc=0, scale=np.sqrt(var[1]), size=boundary_points.shape)
        gaussian_points = np.concatenate((boundary_points + noise_1, boundary_points + noise_2), axis=0)

        # Merge uniform and Gaussian points
        sampled_points = np.concatenate((uniform_points, gaussian_points), axis=0)
        self.sampled_data = self.calculate_sdf(sampled_points)


        # for data that only optimize sdf, we append -100
        # data format: x y sdfValue -100
        self.sampled_data = np.concatenate((self.sampled_data,-100*np.ones((self.sampled_data.shape[0],1))),axis=1)

        # sdf: x y sdf -1000.0
        # gradient: x y grad_x grad_y
        self.sampled_data = np.concatenate((self.sampled_data, grad_data), axis=0)

        # get data for visualizing gradient comparison
        # this is true gradient for sample points
        # data format:x y gx gy  in visual_data
        for i in range(sampled_points.shape[0]):
            cur_visual_piece=sampled_points[i].tolist()+self.shape.get_gradient(sampled_points[i]).tolist()
            #print(cur_visual_piece)
            self.visual_data.append(cur_visual_piece)



        '''
        for i, datum in enumerate(self.sampled_data):
            # 得到梯度方向
            if datum[3]!=-100:
                curGradDirec=datum[2:]
                curCoor=datum[:2]
                curSDF=self.shape.sdf(curCoor)
                #print("Grad ",curGradDirec)
                #print("Coordinate",curCoor)
                goStep=curGradDirec/10000
                #print(goStep)
                thenCoor=curCoor+goStep
                print("then ",thenCoor)
                thenSDF=self.shape.sdf(thenCoor)
                print("then sdf:",thenSDF)
                print("cur sdf:",curSDF)
                if (thenSDF-curSDF)>0:
                    datum=-1*curGradDirec.copy()
                    #print("change dir!")
                else:
                    a=1
        '''

        # data for testing shape and graidient accuracy
        # x y grad_x grad_y
        # x y sdf
        # ============================================================================================================

        test_edge_portion = edge_length / total_length
        test_edge_portion *= tnum
        edge_num = np.around(test_edge_portion).astype(int)

        # Do sampling on edges
        direction = (self.shape.v[1] - self.shape.v[0])
        d = np.random.uniform(1e-7, 1, size=(edge_num[0], 1))
    
        new_boundary_points = self.shape.v[0] + d * direction
        test_normal_direc = np.array([[-direction[1], direction[0]]])

        for i in range(new_boundary_points.shape[0] - 1):
            test_normal_direc = np.concatenate((test_normal_direc, np.array([[-direction[1], direction[0]]])), axis=0)

        test_grad_data = new_boundary_points.copy()
        test_grad_data = np.concatenate((test_grad_data, test_normal_direc), axis=1)

        # get all tesing data 
        for i in range(1, self.shape.num):
            direction = (self.shape.v[(i + 1) % self.shape.num] - self.shape.v[i])
            # print(direction)
            d = np.random.uniform(1e-7, 1, size=(edge_num[i], 1))
       
            cur_points = self.shape.v[i] + d * direction
            if not (cur_points.shape[0] == 0):
                new_boundary_points = np.concatenate((new_boundary_points, self.shape.v[i] + d * direction), axis=0)
                test_normal_direc = np.array([[-direction[1], direction[0]]])

                for j in range(cur_points.shape[0] - 1):
                    test_normal_direc = np.concatenate((test_normal_direc, np.array([[-direction[1], direction[0]]])), axis=0)
     
                cur_grad_data = cur_points.copy()
                cur_grad_data = np.concatenate((cur_grad_data, test_normal_direc), axis=1)
                test_grad_data = np.concatenate((cur_grad_data, test_grad_data), axis=0)
        self.test_accuracy_data=test_grad_data

        # ============================================================================================================

        # Split sampled data into train dataset and val dataset
        train_size = int(len(self.sampled_data) * self.split_ratio)

        choice = np.random.choice(range(self.sampled_data.shape[0]), size=(train_size,), replace=False)
        #print(choice.shape)
        #print(choice)
        ind = np.zeros(self.sampled_data.shape[0], dtype=bool)
        ind[choice] = True
        rest = ~ind
        self.train_data = self.sampled_data[ind]
        self.val_data = self.sampled_data[rest]

    def calculate_sdf(self, points):
        if self.shape.num == 0:
            return

        # Add a third column for storing sdf
        data = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)
        data[:, 2] = np.apply_along_axis(self.shape.sdf, 1, data[:, :2])
        return data

    def save(self):
        if self.shape.num == 0:
            return

        save_name = self.shape_name+"_"+str(SEED_NUM)+"_"+str(SAMPLE_NUM)

        # Save data to .txt
        f = open(f'{self.train_data_path}{save_name}.txt', 'w')
        for datum in self.train_data:
            f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
        f.close()
        f = open(f'{self.val_data_path}{save_name}.txt', 'w')
        for datum in self.val_data:
            f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
        f.close()

        # add files that test the accuracy of the data
        f = open(f'{self.test_accuracy_data_path}{save_name}.txt','w')
        for datum in self.test_accuracy_data:
            f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
        f.close()

        # add files that visualize true gradient of the data
        f = open(f'{self.visual_data_path}{save_name}.txt', 'w')
        for datum in self.visual_data:
            f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
        f.close()

        print(f'Sampled data path = {self.train_data_path}{save_name}.txt\n'
              f'                    {self.val_data_path}{save_name}.txt')

        # Generate a sampled image
        canvas = np.zeros([CANVAS_SIZE[0],CANVAS_SIZE[1],3], np.uint8)
        # Draw polygon
        scaled_v = np.around(self.shape.v * CANVAS_SIZE + CANVAS_SIZE / 2).astype(int)
        cv2.fillPoly(canvas, scaled_v[np.newaxis, :, :], SHAPE_COLOR)
        # Draw points
        for i, datum in enumerate(self.sampled_data):
            point = tuple(np.around(datum[:2] * CANVAS_SIZE + CANVAS_SIZE / 2).astype(int))
            # 得到坐标
            my_p=np.array([datum[0],datum[1]])
            cv2.circle(canvas, point, 1, POINT_COLOR, -1)

            #if i % 50 == 0:
            #    radius = np.abs(np.around(datum[2] * CANVAS_SIZE[0]).astype(int))
            #    cv2.circle(canvas, point, radius, POINT_COLOR)
            '''
            # 得到梯度方向
            if datum[3]!=-100:
                curGradDirec=datum[2:]
                curCoor=datum[:2]
                curSDF=self.shape.sdf(curCoor)
                #print("Grad ",curGradDirec)
                #print("Coordinate",curCoor)
                goStep=curGradDirec/100
                #print(goStep)
                thenCoor=curCoor+goStep
                #print("then ",thenCoor)
                thenSDF=self.shape.sdf(thenCoor)
                #print("then sdf:",thenSDF)
                #print("cur sdf:",curSDF)
                cv2.circle(canvas, tuple(np.around(thenCoor * CANVAS_SIZE + CANVAS_SIZE / 2).astype(int)), 1, POINT_COLOR, -1)
                if (thenSDF-curSDF)>0:
                    datum=-1*curGradDirec.copy()
                    #print("change dir!")
                else:
                    a=1
                    #print("hhh")
            '''

        # 绘制sample点的梯度




        # Store and show
        cv2.imwrite(f'{self.sampled_image_path}{save_name}.png', canvas)
        print(f'Sampled image path = {self.sampled_image_path}{save_name}.png')

        if not self.show_image:
            return

        cv2.imshow('Sampled Image', canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    setup_seed(SEED_NUM)
    print('Enter shape name:')
    #shape_name = input()
    shape_name=SHAPE_NAME
    sampler = ShapeSampler(shape_name, SHAPE_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH,TEST_ACCURACY_DATA_PATH,
                           SAMPLED_IMAGE_PATH, NORM_PATH, NORM_IMAGE_PATH, MASK_PATH, VISUAL_DATA_PATH, show_image=False)
    print('Sampling...')
    sampler.run()
    print('Done!')
