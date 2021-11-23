import torch
from torch_scatter import scatter_add
import numpy as np

def test():
    import numpy as np
    labels = [[401, 260, 1], [427, 270, 1], [441, 280, 1], [434, 290, 1], [412, 300, 1], [390, 310, 1], [368, 320, 1], [347, 330, 1], [325, 340, 1], [303, 350, 1], [277, 360, 1], [247, 370, 1], [216, 380, 1], [185, 390, 1], [154, 400, 1], [124, 410, 1], [94, 420, 1], [64, 430, 1], [34, 440, 1], [4, 450, 1], [507, 270, 2], [521, 280, 2], [530, 290, 2], [539, 300, 2], [539, 310, 2], [538, 320, 2], [537, 330, 2], [536, 340, 2], [534, 350, 2], [530, 360, 2], [521, 370, 2], [512, 380, 2], [504, 390, 2], [495, 400, 2], [486, 410, 2], [478, 420, 2], [469, 430, 2], [460, 440, 2], [452, 450, 2], [443, 460, 2], [434, 470, 2], [426, 480, 2], [417, 490, 2], [408, 500, 2], [400, 510, 2], [391, 520, 2], [382, 530, 2], [374, 540, 2], [365, 550, 2], [355, 560, 2], [346, 570, 2], [337, 580, 2], [328, 590, 2], [318, 600, 2], [309, 610, 2], [300, 620, 2], [291, 630, 2], [282, 640, 2], [272, 650, 2], [263, 660, 2], [254, 670, 2], [245, 680, 2], [236, 690, 2], [226, 700, 2], [217, 710, 2], [709, 320, 3], [729, 330, 3], [748, 340, 3], [764, 350, 3], [780, 360, 3], [795, 370, 3], [811, 380, 3], [827, 390, 3], [842, 400, 3], [855, 410, 3], [868, 420, 3], [881, 430, 3], [894, 440, 3], [907, 450, 3], [920, 460, 3], [933, 470, 3], [946, 480, 3], [959, 490, 3], [972, 500, 3], [985, 510, 3], [999, 520, 3], [1012, 530, 3], [1025, 540, 3], [1039, 550, 3], [1053, 560, 3], [1066, 570, 3], [1080, 580, 3], [1094, 590, 3], [1108, 600, 3], [1122, 610, 3], [1135, 620, 3], [1149, 630, 3], [1163, 640, 3], [1177, 650, 3], [1191, 660, 3], [1205, 670, 3], [1218, 680, 3], [1232, 690, 3], [1246, 700, 3], [1260, 710, 3], [726, 290, 4], [777, 300, 4], [817, 310, 4], [858, 320, 4], [897, 330, 4], [935, 340, 4], [974, 350, 4], [1012, 360, 4], [1050, 370, 4], [1087, 380, 4], [1121, 390, 4], [1155, 400, 4], [1189, 410, 4], [1223, 420, 4], [1257, 430, 4]]
    labels = np.array(labels)
    coffecient = torch.tensor([[0.58348501, -0.79861236, 2.30343866, -0.09976104, -1.22268307, 2.43086767]],
                             dtype=torch.float32)

    #labels_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    print(coffecient.shape)
    loss = hnet_loss(labels, coffecient, 0)


    #with tf.Session() as sess:
    #    _loss = sess.run(loss, feed_dict={labels_tensor:labels})
    #    print(_loss)

def hnet_loss(gt_pts, coefficient, name):
    coefficient_slice = torch.cat((coefficient, (torch.tensor([[1.0]], dtype=torch.float32))), dim=1)
    print(coefficient_slice)
    index = torch.tensor([[0, 1, 2, 4, 5, 7, 8]])
    H = coefficient_slice.new_zeros((1,9), dtype=torch.float32)
    H = scatter_add(coefficient_slice, index, out=H)
    H = torch.reshape(H, (3,3))
    print(H)

    #harus diperbaiki lagi
    gt_pts = torch.tensor(gt_pts, dtype=torch.int32)
    print(gt_pts[:,2])
    lane_mask = torch.where(torch.eq(gt_pts[:,2], torch.tensor(1, dtype=torch.int32)))
    print(lane_mask[0])
    lane_pts = []
    gt_pts=gt_pts.numpy()
    
    for idx in lane_mask[0].numpy():
        lane_pts = np.append(lane_pts, [gt_pts[idx,:]])
    lane_pts = torch.tensor(lane_pts, dtype=torch.int32)
    print(lane_pts)

    
if __name__ == '__main__':
    test()