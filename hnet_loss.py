import torch
import torch.nn as nn
from torch_scatter import scatter_add
import numpy as np

def test():
    import numpy as np
    labels = [[401, 260, 1], [427, 270, 1], [441, 280, 1], [434, 290, 1], [412, 300, 1], [390, 310, 1], [368, 320, 1], [347, 330, 1], [325, 340, 1], [303, 350, 1], [277, 360, 1], [247, 370, 1], [216, 380, 1], [185, 390, 1], [154, 400, 1], [124, 410, 1], [94, 420, 1], [64, 430, 1], [34, 440, 1], [4, 450, 1], [507, 270, 2], [521, 280, 2], [530, 290, 2], [539, 300, 2], [539, 310, 2], [538, 320, 2], [537, 330, 2], [536, 340, 2], [534, 350, 2], [530, 360, 2], [521, 370, 2], [512, 380, 2], [504, 390, 2], [495, 400, 2], [486, 410, 2], [478, 420, 2], [469, 430, 2], [460, 440, 2], [452, 450, 2], [443, 460, 2], [434, 470, 2], [426, 480, 2], [417, 490, 2], [408, 500, 2], [400, 510, 2], [391, 520, 2], [382, 530, 2], [374, 540, 2], [365, 550, 2], [355, 560, 2], [346, 570, 2], [337, 580, 2], [328, 590, 2], [318, 600, 2], [309, 610, 2], [300, 620, 2], [291, 630, 2], [282, 640, 2], [272, 650, 2], [263, 660, 2], [254, 670, 2], [245, 680, 2], [236, 690, 2], [226, 700, 2], [217, 710, 2], [709, 320, 3], [729, 330, 3], [748, 340, 3], [764, 350, 3], [780, 360, 3], [795, 370, 3], [811, 380, 3], [827, 390, 3], [842, 400, 3], [855, 410, 3], [868, 420, 3], [881, 430, 3], [894, 440, 3], [907, 450, 3], [920, 460, 3], [933, 470, 3], [946, 480, 3], [959, 490, 3], [972, 500, 3], [985, 510, 3], [999, 520, 3], [1012, 530, 3], [1025, 540, 3], [1039, 550, 3], [1053, 560, 3], [1066, 570, 3], [1080, 580, 3], [1094, 590, 3], [1108, 600, 3], [1122, 610, 3], [1135, 620, 3], [1149, 630, 3], [1163, 640, 3], [1177, 650, 3], [1191, 660, 3], [1205, 670, 3], [1218, 680, 3], [1232, 690, 3], [1246, 700, 3], [1260, 710, 3], [726, 290, 4], [777, 300, 4], [817, 310, 4], [858, 320, 4], [897, 330, 4], [935, 340, 4], [974, 350, 4], [1012, 360, 4], [1050, 370, 4], [1087, 380, 4], [1121, 390, 4], [1155, 400, 4], [1189, 410, 4], [1223, 420, 4], [1257, 430, 4]]
    labels = np.array(labels)
    coefficient = torch.tensor([[0.58348501, -0.79861236, 2.30343866, -0.09976104, -1.22268307, 2.43086767]],
                             dtype=torch.float32)
    num_lanes = 4
    loss_calc = hnet_loss()
    losses = torch.empty(size=(1,1), dtype=torch.float32)
    for lane_idx in [1,2,3,4]:
        _, lane_loss = loss_calc.loss_calc(lane_idx, labels, coefficient)
        losses = torch.cat((losses, lane_loss))
    
    print(losses)
    mean_loss = torch.mean(losses)
    print(mean_loss)
    



    #with tf.Session() as sess:
    #    _loss = sess.run(loss, feed_dict={labels_tensor:labels})
    #    print(_loss)
class hnet_loss():
    def __init__(self):
        self.mseloss = nn.MSELoss(reduction='mean')
    def loss_calc(self, lane_idx, gt_pts, coefficient):
        coefficient_slice = torch.cat((coefficient, (torch.tensor([[1.0]], dtype=torch.float32))), dim=1)
        #print(coefficient_slice)
        index = torch.tensor([[0, 1, 2, 4, 5, 7, 8]])
        H = coefficient_slice.new_zeros((1,9), dtype=torch.float32)
        H = scatter_add(coefficient_slice, index, out=H)
        H = torch.reshape(H, (3,3))
        print(H)

        #harus diperbaiki lagi
        gt_pts = torch.tensor(gt_pts, dtype=torch.float32)
        #print(gt_pts[:,2])
        lane_mask = torch.where(torch.eq(gt_pts[:,2], torch.tensor(lane_idx, dtype=torch.float32)))
        #print(lane_mask[0])
        lane_pts = []
        print(gt_pts.shape, lane_mask[0].shape)
        
        lane = torch.index_select(gt_pts, 0, lane_mask[0])
        lane = torch.transpose(lane, 0, 1)
        print(lane)
        lane_trans = torch.matmul(H, lane)
        print('lane_trans: {}'.format(lane_trans))

        Y = (lane_trans[1,:] / lane_trans[2,:])
        Y_one = Y.new_ones(size=Y.size(), dtype=torch.float32)
        Y_edited = torch.stack((torch.pow(Y,3), torch.pow(Y,2), Y, Y_one), dim=1)
        print('Y_edited: {}'.format(Y_edited))
        Y_edited_T = torch.transpose(Y_edited,0 ,1)
        X = (lane_trans[0,:] / lane_trans[2,:]).unsqueeze(1)
        print('X: {} '.format(X))

        w = torch.matmul(torch.matmul(torch.inverse(torch.matmul(Y_edited_T, Y_edited)),Y_edited_T), X)
        print('w: {}'.format(w))

        x_pred = torch.matmul(Y_edited, w)
        print('x_pred: {}'.format(x_pred))

        lane_trans_pred = torch.transpose(torch.stack([torch.squeeze(x_pred, dim=-1) * lane_trans[2, :],
                                                        Y * lane_trans[2, :], lane_trans[2, :]], dim=1), 0, 1
        )
        print('lane_trans_pred: {}'.format(lane_trans_pred))

        lane_trans_back = torch.matmul(torch.inverse(H), lane_trans_pred)
        #print(lane_trans_back)

        #print(lane[0,:])
        #print(lane_trans_back[0,:])

        loss = self.mseloss(lane[0,:], lane_trans_back[0,:])
        loss = loss.unsqueeze(0).unsqueeze(0)
        print(loss)
        
        return H, loss

if __name__ == '__main__':
    test()