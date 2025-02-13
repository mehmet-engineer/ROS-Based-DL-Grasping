import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from .grasp import GraspRectangles, detect_grasps

#matplotlib.use("TkAgg")
def plot_output(rgb_img,rgb_img_1, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None,
                grasp_q_img_ggcnn=None,grasp_angle_img_ggcnn=None,grasp_width_img_ggcnn=None):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(3, 3, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    # ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(3, 3, 2)
    # ax.imshow(depth_img, cmap='gist_rainbow')
    ax.imshow(depth_img,)
    for g in gs:
        g.plot(ax)
    ax.set_title('Depth')
    ax.axis('off')

    ax = fig.add_subplot(3, 3, 3)
    # plot = ax.imshow(grasp_width_img, cmap='prism', vmin=-0, vmax=150)
    # ax.set_title('q image')

    ax = fig.add_subplot(3, 3, 4)
    plot = ax.imshow(grasp_q_img, cmap="terrain", vmin=0, vmax=1)  #？terrain
    ax.axis('off')
    ax.set_title('q image')


    ax = fig.add_subplot(3, 3, 5)  #flag  prism jet
    plot = ax.imshow(grasp_angle_img, cmap="prism", vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.axis('off')
    ax.set_title('angle')

    ax = fig.add_subplot(3, 3, 6,)
    plot = ax.imshow(grasp_width_img, cmap='hsv', vmin=-0, vmax=150)
    ax.set_title('width')
    ax.axis('off')



    ax = fig.add_subplot(3, 3, 7, )
    plot = ax.imshow(grasp_q_img_ggcnn, cmap='terrain', vmin=0, vmax=1)
    ax.set_title('q image')
    ax.axis('off')

    ax = fig.add_subplot(3, 3, 8)  # flag  prism jet
    plot = ax.imshow(grasp_angle_img_ggcnn, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.axis('off')
    ax.set_title('angle')

    ax = fig.add_subplot(3, 3, 9, )
    plot = ax.imshow(grasp_width_img_ggcnn, cmap='hsv', vmin=-0, vmax=150)
    ax.set_title('width')
    ax.axis('off')
    # for g in gs:
    #     g.plot(ax)
    # ax.set_title('Angle')

    # plt.colorbar(plot)

    # plt.imshow(rgb_img)
    plt.show()

    if (input())==str(1):
       plt.figure(figsize=(5,5))
       plt.imshow(rgb_img)
       for g in gs:
           g.plot(plt)
       plt.axis("off")
       plt.tight_layout()
       plt.show()

       plt.figure(figsize=(5, 5))
       # plt.imshow(depth_img, cmap='gist_gray')
       plt.imshow(depth_img, )
       plt.axis("off")
       for g in gs:
           g.plot(plt)
       plt.tight_layout()
       plt.show()

       plt.figure(figsize=(5, 5))
       plt.imshow(grasp_q_img, cmap="terrain", vmin=0, vmax=1)
       plt.axis("off")
       plt.tight_layout()
       plt.show()

       plt.figure(figsize=(5, 5))
       plt.imshow(grasp_angle_img, cmap="prism", vmin=-np.pi / 2, vmax=np.pi / 2)
       plt.axis("off")
       plt.tight_layout()
       plt.show()

       plt.figure(figsize=(5, 5))
       plt.imshow(grasp_width_img, cmap='hsv', vmin=-0, vmax=150)
       plt.axis("off")
       plt.tight_layout()
       plt.show()


       plt.figure(figsize=(5, 5))
       plt.imshow(grasp_q_img_ggcnn, cmap="terrain", vmin=0, vmax=1)
       plt.axis("off")
       plt.tight_layout()
       plt.show()

       plt.figure(figsize=(5, 5))
       plt.imshow(grasp_angle_img_ggcnn, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
       plt.axis("off")
       plt.tight_layout()
       plt.show()

       plt.figure(figsize=(5, 5))
       plt.imshow(grasp_width_img_ggcnn, cmap='hsv', vmin=-0, vmax=150)
       plt.axis("off")
       plt.tight_layout()
       plt.show()
def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, threshold=0.25):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > threshold:
            return True
    else:
        return False