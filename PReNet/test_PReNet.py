import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from PReNet.utils import *
from PReNet.networks import *
import time 

def Predict(dataRoot, logRoot, savepath, use_GPU=True):
    # parser = argparse.ArgumentParser(description="PReNet_Test")
    # parser.add_argument("--logdir", type=str, default=logRoot, help='path to model and log files')
    # parser.add_argument("--data_path", type=str, default=dataRoot, help='path to training data')
    # parser.add_argument("--save_path", type=str, default=savepath, help='path to save results')
    # parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
    # parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
    # parser.add_argument("--recurrent_iter", type=int, default=4, help='number of recursive stages')
    # opt = parser.parse_args()

    if use_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    os.makedirs(savepath, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = PReNet(4, use_GPU)
    print_network(model)
    if use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(logRoot, 'net_latest.pth')))
    model.eval()

    time_test = 0
    count = 0
    for img_name in os.listdir(dataRoot):
        if is_image(img_name):
            img_path = os.path.join(dataRoot, img_name)

            # input image
            y1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            y = cv2.resize(y1, (908, 512), interpolation=cv2.INTER_CUBIC)
            h, w = y.shape[:2]
            # y = y[int(h // 2 * (1 - 1 / 2 ** 0.5)):int(h // 2 * (1 + 1 / 2 ** 0.5)),
            #          int(w // 2 * (1 - 1 / 2 ** 0.5)):int(w // 2 * (1 + 1 / 2 ** 0.5))]

            # b, g, r = cv2.split(y)
            # y = cv2.merge([r, g, b])
            #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)
            edge = cv2.Canny(y, 50, 150)


            y = normalize(np.float32(y))
            y = np.expand_dims(y, 0)
            y = Variable(torch.Tensor(y))
            y = y.resize(y.shape[0], 1, y.shape[1], y.shape[2])

            edge = normalize(np.float32(edge))
            edge = np.expand_dims(edge, 0)
            edge = Variable(torch.Tensor(edge))
            edge = edge.resize(edge.shape[0], 1, edge.shape[1], edge.shape[2])


            y = torch.cat((y, edge), 1)


            if use_GPU:
                y = y.cuda()

            with torch.no_grad(): #
                if use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()

                out, _ = model(y)
                out = torch.clamp(out, 0., 1.)


                if use_GPU:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name, ': ', dur_time)

            if use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())
            save_out = np.array([save_out])
            #print(save_out.shape)
            save_out = save_out.transpose(1, 2, 0)
            # b, g, r = cv2.split(save_out)
            # save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(savepath, img_name[:-4]+'.png'), save_out)

            count += 1

    print('Avg. time:', time_test/count)

#
# if __name__ == "__main__":
#     main()

