import random
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import cv2
import numpy as np


def safe_collate(batch):
    """Return batch without any None values"""
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class CocoDataset(Dataset):
    def __init__(self, dataset_dir, subset, year="2014",
                 patch_size=128, rho=32, img_h=240, img_w=320):
        super(CocoDataset, self).__init__()
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)
        self.fnames = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.patch_size = patch_size
        self.rho = rho
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        image = cv2.imread(self.fnames[index], 0)

        image = cv2.resize(image, (self.img_w, self.img_h))
        height, width = image.shape

        # create random point P within appropriate bounds
        y = random.randint(self.rho, height - self.rho - self.patch_size)
        x = random.randint(self.rho, width - self.rho - self.patch_size)
        # define corners of image patch
        top_left_point = (x, y)
        bottom_left_point = (x, self.patch_size + y)
        bottom_right_point = (self.patch_size + x, self.patch_size + y)
        top_right_point = (x + self.patch_size, y)

        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append((point[0] + random.randint(-self.rho, self.rho),
                                          point[1] + random.randint(-self.rho, self.rho)))

        y_grid, x_grid = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

        # Two branches. The CNN try to learn the H and inv(H) at the same time. So in the first branch, we just compute the
        #  homography H from the original image to a perturbed image. In the second branch, we just compute the inv(H)
        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
        try:
            H_inverse = np.linalg.inv(H)
        except:
            # either matrix could not be solved or inverted
            # this will show up as None, so use safe_collate in train.py
            return

        warped_image = cv2.warpPerspective(image, H_inverse, (image.shape[1], image.shape[0]))

        img_patch_ori = image[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]]
        img_patch_pert = warped_image[top_left_point[1]:bottom_right_point[1],
                                      top_left_point[0]:bottom_right_point[0]]

        point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float32), H).squeeze()
        diff_branch1 = point_transformed_branch1 - point
        diff_x_branch1 = diff_branch1[:, 0]
        diff_y_branch1 = diff_branch1[:, 1]

        diff_x_branch1 = diff_x_branch1.reshape((image.shape[0], image.shape[1]))
        diff_y_branch1 = diff_y_branch1.reshape((image.shape[0], image.shape[1]))

        pf_patch_x_branch1 = diff_x_branch1[top_left_point[1]:bottom_right_point[1],
                                            top_left_point[0]:bottom_right_point[0]]

        pf_patch_y_branch1 = diff_y_branch1[top_left_point[1]:bottom_right_point[1],
                                            top_left_point[0]:bottom_right_point[0]]

        pf_patch = np.zeros((2, self.patch_size, self.patch_size))
        pf_patch[0, :, :] = pf_patch_x_branch1
        pf_patch[1, :, :] = pf_patch_y_branch1

        img_patch_ori = img_patch_ori / 255
        img_patch_pert = img_patch_pert / 255
        image_patch_pair = np.zeros((2, self.patch_size, self.patch_size))
        image_patch_pair[0, :, :] = img_patch_ori
        image_patch_pair[1, :, :] = img_patch_pert

        return image_patch_pair, pf_patch

class ImageDataset(Dataset):
    def __init__(self, dataset_dir, subset, rho=32, img_h=240, img_w=320):
        super(ImageDataset, self).__init__()
        image_dir = "{}/{}".format(dataset_dir, subset)
        self.fnames = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.rho = rho
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        image = cv2.imread(self.fnames[index], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_w, self.img_h))
        height, width = image.shape

        #против часовой стрелки
        top_left_point = (0, 0)
        bottom_left_point = (0, height)
        bottom_right_point = (width, height)
        top_right_point = (width, 0)
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append((point[0] + random.randint(-self.rho, self.rho),
                                          point[1] + random.randint(-self.rho, self.rho)))


        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))

        H_inverse = np.linalg.inv(H)

        warped_image = cv2.warpPerspective(image, H_inverse, (image.shape[1], image.shape[0]))

        y_grid, x_grid = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

        point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float32), H).squeeze()
        diff_branch1 = point_transformed_branch1 - point
        diff_x_branch1 = diff_branch1[:, 0]
        diff_y_branch1 = diff_branch1[:, 1]

        diff_x_branch1 = diff_x_branch1.reshape((image.shape[0], image.shape[1]))
        diff_y_branch1 = diff_y_branch1.reshape((image.shape[0], image.shape[1]))

        pf_patch = np.zeros((2, image.shape[0], image.shape[1]))
        pf_patch[0, :, :] = diff_x_branch1
        pf_patch[1, :, :] = diff_y_branch1

        img_patch_ori = image / 255
        img_patch_pert = warped_image / 255
        image_patch_pair = np.zeros((2, image.shape[0], image.shape[1]))
        image_patch_pair[0, :, :] = img_patch_ori
        image_patch_pair[1, :, :] = img_patch_pert

        return image_patch_pair, pf_patch, H

if __name__ == "__main__":
    folder_images = "/home/gamma/ABT_WORK/khl/video_stab/images"
    dataset = ImageDataset(folder_images, rho=64, img_h=360, img_w=640)
    random_idx = np.random.randint(len(dataset), size=10)
    print("len of dataset: ", len(dataset))
    folder_samples = "/home/gamma/MY_WORK/PFNet-pytorch/samples"
    for idx in random_idx:
        img_pair, pf_patch, H = dataset[idx]
        img_ori = img_pair[0] * 255
        img_pert = img_pair[1] * 255
        img_samples = np.concatenate([img_ori, img_pert], axis=1)
        cv2.imwrite(os.path.join(folder_samples, "{}.jpg".format(idx)), img_samples)
