import cv2
import matplotlib.pyplot as plt
img=cv2.imread("C:\\Users\soula\\Image-Super-Resolution-via-Iterative-Refinement\\experiments\\distributed_high_sr_230715_164017\\results\\0_1_sr.tif",cv2.IMREAD_ANYDEPTH)

# Display the image in a window
image = img.astype('float32')
plt.imshow(img)
plt.axis('off')  # Remove the axes
plt.show()



# import cv2

# def select_roi(image):
#     # Check if image is loaded successfully
#     if image is None:
#         print("Failed to load image")
#         return None

#     # Display the image and allow the user to select the ROI
#     cv2.imshow('Image', image)
#     bbox = cv2.selectROI('Image', image, fromCenter=False, showCrosshair=True)

#     # Crop the image based on the selected ROI
#     x, y, width, height = bbox
#     cropped_image = image[y:y+height, x:x+width]
#     print(width,height)
#     print("x",x)
#     print("y",y)
#     # Close the image window
#     cv2.destroyAllWindows()

#     return cropped_image

# # Example usage
# image = cv2.imread("C:\\Users\\soula\\Downloads\\013395_0 (4).png")

# # Select the ROI manually
# cropped_image = select_roi(image)

# # Display the cropped image if it is not None
# if cropped_image is not None:
#     cv2.imshow('Cropped Image', cropped_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
