import numpy as np
import torch


# The Geometry and Polygon classes are adapted from
# https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb
class Geometry(object):
    EPS = 1e-12

    @staticmethod
    def distance_from_point_to_segment(a, b, p):
        res = min(np.linalg.norm(a - p), np.linalg.norm(b - p))
        if (np.linalg.norm(a - b) > Geometry.EPS
                and np.dot(p - a, b - a) > Geometry.EPS
                and np.dot(p - b, a - b) > Geometry.EPS):
            res = abs(np.cross(p - a, b - a) / np.linalg.norm(b - a))
        return res


class Polygon(object):
    def __init__(self):
        self.v = np.array([])
        # Number of vertices/edges
        self.num = 0

    def set_v(self, v):
        self.v = v
        self.num = len(self.v)

    def sdf(self, p):
        return -(self.distance(p))["distance"] if self.inside(p) else (self.distance(p))["distance"]

    def inside(self, p):
        angle_sum = 0
        for i in range(self.num):
            a = self.v[i]
            b = self.v[(i + 1) % self.num]
            angle_sum += np.arctan2(np.cross(a - p, b - p), np.dot(a - p, b - p))
        return abs(angle_sum) > 1

    # 得到sdf
    def distance(self, p):
        #p=torch.tensor(p,requires_grad=True)

        res = Geometry.distance_from_point_to_segment(self.v[-1], self.v[0], p)
        dis1 = np.linalg.norm(self.v[-1] - p)
        dis2 = np.linalg.norm(self.v[0] - p)
        if res == dis1:
            res_dir = -self.v[-1] + p
        elif res == dis2:
            res_dir = -self.v[0] + p
        else:
            res_dir = self.v[0] - self.v[1]
            res_dir = [res_dir[1], -res_dir[0]]
            res_dir = np.array(res_dir)


        for i in range(len(self.v) - 1):
            cur_res=Geometry.distance_from_point_to_segment(self.v[i], self.v[i + 1], p)
            if res > cur_res:
                dis1=np.linalg.norm(self.v[i]-p)
                dis2=np.linalg.norm(self.v[i+1]-p)
                if cur_res==dis1:
                    res_dir = -self.v[i] + p
                elif cur_res==dis2:
                    res_dir = -self.v[i+1]+p
                else:
                    res_dir = self.v[i] - self.v[i+1]
                    res_dir = [res_dir[1], -res_dir[0]]
                    res_dir = np.array(res_dir)
                res=cur_res


        res_dir=res_dir/np.linalg.norm(res_dir)

        return {"distance":res,
                "gradient":res_dir
                }

    # get normalized gradient
    def get_gradient(self,p):
        raw_grad=self.distance(p)["gradient"]
        cur_sdf=self.sdf(p)
        go_step=raw_grad*0.0001
        after_p=p+go_step
        after_sdf=self.sdf(after_p)
        if after_sdf<cur_sdf:
            raw_grad=-raw_grad
        return raw_grad

    def load(self, path, name):
        vertices = []
        f = open(f'{path}{name}.txt', 'r')
        line = f.readline()
        while line:
            x, y = map(lambda n: np.double(n), line.strip('\n').split(' '))
            vertices.append([x, y])
            line = f.readline()
        f.close()
        self.set_v(np.array(vertices, dtype=np.double))
