# Q3_graded
# Do not change the above line.

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img


def load_images():
    patterns = []
    images = ['1.jpg', '2.jpg', '3.png', '4.jpg']
    for i in images:
        img = load_img(i, color_mode="grayscale")
        patterns.append(img_to_array(img).reshape(-1))
    return patterns

# Q3_graded
# Do not change the above line.

patterns = load_images()
def normalize(patterns):
    p_copy = []
    for p in patterns:
        p_copy.append(np.where(p <= 210, 1, -1).reshape(-1)) # convert pixel with value smaller than 210 to 1 and others to -1
    return p_copy
normalized_patterns = normalize(patterns)

def convert_to_img(patterns):
    p_copy = []
    for p in patterns:
        p_copy.append(np.where(p == -1, 255, 0)) # convert -1 to 255 and 1 to 0
    return p_copy

# Q3_graded
# Do not change the above line.

def calculate_weights(patterns, size):
    weights = np.zeros((size, size))
    for p in patterns:
        p = p.reshape((-1, 1))
        weights += p.dot(p.T)
    np.fill_diagonal(weights, 0)
    return weights

weights = calculate_weights(normalized_patterns, 100*100)

# Q3_graded
# Do not change the above line.

def next_pattern(pattern, weights):
    return np.sign(np.dot(pattern, weights))

def calculate_accuracy(pattern, restored):
    return np.sum(pattern == restored) / restored.shape[0]

# Q3_graded
# Do not change the above line.

test_img_1 = load_img('test1.jpg', color_mode='grayscale')
test_img_1 = img_to_array(test_img_1)
z = np.zeros((50, 100, 1)) + 255
test_img_1_resized = np.vstack((z, test_img_1))

test_img_2 = load_img('test2.jpg', color_mode='grayscale')
test_img_2 = img_to_array(test_img_2)
z = np.zeros((100, 100, 1)) + 255
z[:50, :50] = test_img_2
test_img_2_resized = z

test_img_3 = load_img('test3.png', color_mode='grayscale')
test_img_3 = img_to_array(test_img_3)
z = np.zeros((100, 100, 1)) + 255
z[:, 50:] = test_img_3
test_img_3_resized = z


test_img_4 = load_img('test4.jpg', color_mode='grayscale')
test_img_4 = img_to_array(test_img_4)
z = np.zeros((100, 100, 1)) + 255
z[30:70, 20:60] = test_img_4
test_img_4_resized = z
array_to_img(test_img_4_resized.reshape((100,100,-1)))


normalized_test_images = normalize([test_img_1_resized, test_img_2_resized, test_img_3_resized, test_img_4_resized])

# Q3_graded
# Do not change the above line.

restored_1 = next_pattern(normalized_test_images[0], weights)
restored_img_1 = convert_to_img([restored_1])
print('accuracy is', calculate_accuracy(normalized_patterns[0], restored_1))
array_to_img(restored_img_1[0].reshape((100, 100, -1)))

# Q3_graded
# Do not change the above line.

restored_2 = next_pattern(normalized_test_images[1], weights)
restored_img_2 = convert_to_img([restored_2])
print('accuracy is', calculate_accuracy(normalized_patterns[1], restored_2))
array_to_img(restored_img_2[0].reshape((100, 100, -1)))

# Q3_graded
# Do not change the above line.

restored_3 = next_pattern(normalized_test_images[2], weights)
restored_img_3 = convert_to_img([restored_3])
print('accuracy is', calculate_accuracy(normalized_patterns[2], restored_3))
array_to_img(restored_img_3[0].reshape((100, 100, -1)))

# Q3_graded
# Do not change the above line.

restored_4 = next_pattern(normalized_test_images[3], weights)
restored_img_4 = convert_to_img([restored_4])
print('accuracy is', calculate_accuracy(normalized_patterns[3], restored_4))
array_to_img(restored_img_4[0].reshape((100, 100, -1)))

