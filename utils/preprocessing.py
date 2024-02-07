"""
Authors: Oshani Jayawardane, Buddika Weerasinghe

This file contains the functions for preprocessing the mammogram images. The following functions are included:
1. crop_breast(image, mask): 
        - Crop the breast region of the mammogram image
        - returns the cropped image, cropped mask, new height, new width
        
2. dilate(img, kernel_size, iterations): 
        - Dilate the image
        - returns the dilated image

3. truncate(img, lower_percentile, upper_percentile): 
        - Truncate the image using percentile values and clip the values above and below the thresholds
        - returns the truncated image

4. clahe(img, clip, gridSize): 
        - Apply CLAHE for contrast enhancement
        - returns the enhanced image

5. normalize_min_max(img): 
        - Normalize the image using min-max normalization
        - returns the normalized image

6. normalize_standardization(img, mean, std): 
        - Normalize the image using standardization
        - returns the normalized image

7. normalize_constant_scaling(img): 
        - Normalize the image using constant scaling
        - returns the normalized image

The functions except the images and masks to be in *** grayscale format *** (single channel).

"""


import cv2
import numpy as np

#################################################
# Crop the breast ROI
#################################################

def crop_breast(image, mask):
    """
    @img : numpy array image
    @mask: numpy array mask
    return: numpy array of the cropped ROI
    """

    image = image[5:-5, 5:-5]
    
    _, thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    height, width = thresh.shape
    
    # look for lines of 255 pixels that stretch from width = 0 to width = width and replace that entire line by 0
    for y in range(height):
        if all(thresh[y, :] == 255):
            thresh[y, :] = 0
    
    # look for lines of 255 pixels that stretch from height = 0 to height = height and replace that entire line by 0       
    for x in range(width):
        if all(thresh[x, :] == 255):
            thresh[x, :] = 0
            
    inter = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    out = np.zeros(image.shape, np.uint8)
    cv2.drawContours(out, [largest_contour], -1, 255, cv2.FILLED)
    out = cv2.bitwise_and(image, out)
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Dynamic padding adjustment
    padding=50
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2*padding, image.shape[1] - x)
    h = min(h + 2*padding, image.shape[0] - y)
    cropped_image = out[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    new_height, new_width = cropped_image.shape[:2]
    
    return cropped_image, cropped_mask, new_height, new_width


#################################################
# Dilate Image
#################################################

def dilate(img, kernel_size, iterations):
    """
    @img : numpy array image
    return: numpy array of denoised image
    """
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=iterations)
    
    return dilation


#################################################
# Truncate Image
#################################################

def truncate(img, lower_percentile, upper_percentile):
    """
    @img : numpy array image
    @lower_percentile : float, lower percentile value (0-100)
    @upper_percentile : float, upper percentile value (0-100)
    return: numpy array of the truncated image
    """
    lower_threshold = np.percentile(img, lower_percentile)
    upper_threshold = np.percentile(img, upper_percentile)
    
    truncated_image = np.clip(img, lower_threshold, upper_threshold)
    
    return truncated_image.astype('uint8')


#################################################
# CLAHE for contrast enhancement
#################################################

def clahe(img, clip, gridSize):
    """
    @img : numpy array image
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(gridSize, gridSize))
    enhanced_image = clahe.apply(img)
    
    return enhanced_image


#################################################
# Normalize Image - 3 different methods
#################################################

def normalize_min_max(img):
    """
    @img : numpy array image
    return: numpy array of the normalized image
    """
    min_val = np.min(img)
    max_val = np.max(img)
    normalized = (img - min_val)/(max_val - min_val)
    
    return normalized


def normalize_standardization(img, mean, std):
    """
    @img : numpy array image
    return: numpy array of the normalized image
    """
    normalized = (img - mean)/std

    return normalized


def normalize_constant_scaling(img):
    """
    @img : numpy array image
    return: numpy array of the normalized image
    """
    normalized = img.astype('float32') / 255.0

    return normalized