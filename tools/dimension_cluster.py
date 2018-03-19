import numpy as np
#from lib.utils import *
import glob

class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def int_left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return (int(round(self.x - half_width)), int(round(self.y - half_height)))

    def left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return [self.x - half_width, self.y - half_height]

    def int_right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return (int(round(self.x + half_width)), int(round(self.y + half_height)))

    def right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return [self.x + half_width, self.y + half_height]

    def crop_region(self, h, w):
        left, top = self.left_top()
        right, bottom = self.right_bottom()
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)
        self.w = right - left
        self.h = bottom - top
        self.x = (right + left) / 2
        self.y = (bottom + top) / 2
        return self
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u
def overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left
# hyper parameters
label_path = "../data/VIVAdevkit/Labels/"
n_anchors = 9
loss_convergence = 1e-5
image_width = 960
image_height = 1280
grid_width = 16
grid_height = 16

boxes = []
label_files = glob.glob("%s/*.txt" % label_path)
for label_file in label_files:
    with open(label_file, "r") as f:
        label, xmin, ymin, xmax, ymax = f.read().strip().split(" ")[1:6]
	w = float(ymax) - float(ymin)
	h = float(xmax) - float(xmin)
        boxes.append(Box(0, 0, float(w), float(h)))

# initial centroids
centroid_indices = np.random.choice(len(boxes), n_anchors)
centroids = []
for centroid_index in centroid_indices:
    centroids.append(boxes[centroid_index])

# do k-means
def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        # print 'i=',i, 'len = ',len(groups[i])
        if len(groups[i]) == 0: continue
        new_centroids[i].w /= float(len(groups[i]))
        new_centroids[i].h /= float(len(groups[i]))

    return new_centroids, groups, loss

# iterate k-means
new_centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
while(True):
    new_centroids, groups, loss = do_kmeans(n_anchors, boxes, new_centroids)
    print("loss = %f" % loss)
    if abs(old_loss - loss) < loss_convergence:
        break
    old_loss = loss

# print result
for centroid in centroids:
    print(centroid.w, centroid.h)
