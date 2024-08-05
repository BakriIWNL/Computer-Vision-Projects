import numpy as np
import cv2
from PIL import Image
import sys
import matplotlib.pyplot as plt

def recvieve_image(shredded_array):
    
    #given shred width
    shred_width = 64
    # Convert the image to grayscale
    shredded_array = Image.fromarray(shredded_array)
    shredded_array = shredded_array.convert('L')
    shredded_array = np.array(shredded_array)

    # Get height of the shred and number of shreds, 
    # shredded_array.shape[1] is the width of the image divided by shred_width (64)
    shreds_count = shredded_array.shape[1]//shred_width
    h_shred = shredded_array.shape[0]
    shred = []

    # find the shreds, by taking all the rows and the columns from i*shred_width to (i+1)*shred_width
    for i in range(shreds_count):
        shred.append(shredded_array[:,i*shred_width:(i+1)*shred_width])

    # Calculate the difference between each shred and every other shred
    # Compare right side of each shred with left side of every other shred
    diff = np.zeros((shreds_count, shreds_count))
    for i in range(shreds_count):
        for j in range(shreds_count):
            # sum of squares of differences
            diff[i, j] = np.sum((shred[i][:, shred_width-1] - shred[j][:, 0]) ** 2)
            # print(i, j,diff[i, j])

    # Find the best match for each shred
    shred_order = [0]
    for i in range(1, shreds_count):  # Exclude shred 0 from the loop
        last_shred = shred_order[-1]
        best_match = np.argmin(diff[last_shred])
        min_diff = diff[last_shred, best_match]

        # Check if any other shreds have a lower diff with the best_match shred
        for j in range(shreds_count):
            if j not in shred_order and diff[j, best_match] < min_diff:
                min_diff = diff[j, best_match]
                best_match = j

        # If the best match is already in the shred_order, find the next best match
        while best_match in shred_order:
            diff[last_shred, best_match] = sys.maxsize  # Set its difference to maxsize
            best_match = np.argmin(diff[last_shred])  # And find the next best match

        shred_order.append(best_match)
        diff[:, best_match] = sys.maxsize

        # Display the shred that is being chosen right now
        # print(last_shred, best_match, diff[last_shred, best_match])
        # print(diff[2, 10],diff[4,10])

    # shred_order = [0, 11, 7, 1, 8, 9, 3, 5, 6, 4, 2, 10]
    # Reorder the shreds and reconstruct the image
    shredded_image = np.zeros((h_shred, shreds_count * shred_width))
    for i in range(shreds_count):
        shredded_image[:, i * shred_width:(i + 1) * shred_width] = shred[shred_order[i]]

    # display image
    shredded_image = shredded_image.astype(np.uint8)
    plt.imshow(shredded_image, cmap='gray')
    plt.show()

    # Output the new order of shreds
    print(shred_order)

    return shred_order

image = cv2.imread('shredded.jpg', cv2.IMREAD_GRAYSCALE)
shredded_array = np.array(image)
shred_order = recvieve_image(shredded_array)
# print(shred_order)