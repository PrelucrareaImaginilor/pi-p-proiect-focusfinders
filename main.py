import os

import cv2
import matplotlib.pyplot as plt
import glob
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)


orb = cv2.ORB_create(nfeatures=1000)

image_files = glob.glob('dataSet/*.png')

keypoints_list = []
descriptors_list = []

for img_file in image_files:
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches_list = []
    for i in range(len(descriptors_list) - 1):
        matches = bf.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        matches_list.append(good_matches)

        img1 = cv2.imread(image_files[i])
        img2 = cv2.imread(image_files[i + 1])
        img_matches = cv2.drawMatches(img1, keypoints_list[i], img2, keypoints_list[i + 1], good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(10, 10))
        plt.imshow(img_matches)
        plt.title(f"Potriviri între {i + 1} și  {i + 2}")
        plt.axis('off')

        plt.savefig(os.path.join(output_dir, f'matches_{i + 1}_{i + 2}.png'), bbox_inches='tight')
        plt.close()

        print(f"Numărul de potriviri bune între imaginea {i + 1} și imaginea {i + 2}: {len(good_matches)}")






