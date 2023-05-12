import numpy as np
import cv2
from PIL import Image
import os
import csv

######### Declare inputs and variables #####################
distanceMetrices = {0 : "HELLINGER", 1 : "HISTOGRAM_DISTANCE"}
DISTANCE_METRIC = 0 

CELL_SIZE = 8
STEP_SIZE = 8

output_dir = "Output"  ## directory to store outputs
dir_feature_files = "Features" ## directory to store hog features ASCII .txt files
dir_gradient_imgs = "Gradients" ## directory to store gradient images

path_feature_dir = os.path.join(output_dir, dir_feature_files)
path_grad_dir = os.path.join(output_dir, dir_gradient_imgs)


filter_sobel_hor = np.array([[-1,-2,-1],
                             [0,0,0],
                             [1,2,1]])

filter_sobel_ver = np.array([[-1,0,1],
                             [-2,0,2],
                             [-1,0,1]])


def convolution(imgInput, kernel):
    """Function to perform convolution on the image.
    Param : imgInput - input image (numpy array)
    kernel : filter / kernel (numpy array)
    Return : imgWithConvolution (numpy array)
    """
    ## get image size
    image_h = imgInput.shape[0]
    image_w = imgInput.shape[1]
    
    ## getting kernel shape    
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    
    ## Calculate kernel half sizes for accomodating the border where kernel goes out of image bourndries
    h = kernel_h // 2
    w = kernel_w // 2
    
    ## Initialize output image with zero values
    image_conv = np.zeros(imgInput.shape)
    
    ## Perform convolution
    for i in range(h, image_h-h):        
        for j in range(w, image_w-w):
            x = imgInput[i-h:i-h+kernel_h, j-w:j-w+kernel_w]
            # multiply with the kernel
            
            x = x.flatten()*kernel.flatten()
            # assign mulitplied output of kernel to the pixel
            # print(x)
            image_conv[i][j] = x.sum()

    ## At the boundries where kernel goes out of image, set values as undefineed i.e. np.NaN
    image_conv[:h,:] = 0
    image_conv[image_h-h:,:] = 0
    image_conv[:,:w] = 0
    image_conv[:,image_w-w:] = 0
    
    
    return image_conv


def convert_to_gray(img):
    """funtion to conver color image array to gray scale image"""
    #Split b, g, r channels
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = r * 0.299 + g * 0.587 +  b * 0.114
    ## Round elements to nearest integers
    gray_new = np.rint(gray)
    ## Convert to int8 data type
    gray_new = gray_new.astype(np.int8)
    return gray_new




def normalize_magnitude(mag):
    """Funtion to normalize the magnitude"""
    ## 1. Normalize gradient magnitude within range 0 to 255
    mag_normed = mag / np.max(mag) * 255

    ## 2. Convert to integers (round off)
    mag_normed = mag_normed.astype(np.uint8)
    return mag_normed

def quantize_unsigned(angle):
    """quantization of unsigned angles"""        
    #### Subtract 180 if angle is within range [180,360)
    if angle > 180:  
        angle = angle - 180
    upperMargins = [20,40,60,80,100,120,140,160,180]
    bin_index = 8
    for i, upper in enumerate(upperMargins):
        if angle <upper:
            bin_index = i
            break
    return bin_index

def quantize_signed(angle, no_of_bins=9):
    """quantization of signed angles"""        
    bin_size = 360 / no_of_bins
    bin_index = int((angle+180) //bin_size)
    return bin_index

def create_grad_array(img):
    """Function that returns gradient and magnitude using Sobel operator"""
    ## 1. Convert img to Gray
    grayImg = convert_to_gray(img)
    ## 2. Apply horizontal and vertical sobel operator convolution
    Gx = convolution(grayImg, filter_sobel_hor)
    Gy = convolution(grayImg, filter_sobel_ver)
    ## 3. Calculate magnitude and angles from gradients
    mag = (Gx**2 + Gy**2)**(1/2)
    grad_angle = np.arctan2(Gy,Gx)*(180/np.pi)

    return mag, grad_angle  

# Create histogram of gradients
def get_cell_histogram(quantized_angles, mag_cell, gradient_angles_cell):
    """Function to return Histogram of a single 8x8 cell
    Params : quantized_angles -  quantized angle array
    mag_cell - magnitude array (of cell)  
    gradient_angles_cell - gradient array (of cell)
    
    """
    num_bins = 9
    histogram = np.zeros(num_bins, dtype=float)
    bin_centers = [10,30,50,70,90,110,130,150,170]

    for i, bin_index in enumerate(quantized_angles):
        angle = gradient_angles_cell[i]
        
        resp_weight = mag_cell[i]
        for j, center in enumerate(bin_centers):
            if j==0:
                ## If angle is <= 10 i.e last bin center add complete weight to first bin
                if angle <= bin_centers[j]:
                    histogram[0] +=resp_weight
                    break
            else:
                if angle < bin_centers[j] and angle>= bin_centers[j-1]:
                    dist_prev = angle - bin_centers[j-1]
                    dist_next = bin_centers[j] - angle
                    w0 = resp_weight * (dist_prev / 20)
                    w1 = resp_weight * (dist_next / 20)
                    # print(f"bin_centers[j-1] {bin_centers[j-1]} bin_centers[j] {bin_centers[j]} dist_prev {dist_prev} dist_next {dist_next} w0 {w0} w1 {w1}")

                    histogram[j-1] +=w0
                    histogram[j] +=w1
                    # print(histogram[j-1], histogram[j])

                    break
                
            if j ==8:
                ## If angle is > 170 i.e last bin center add complete weight to last bin 
                if angle>=bin_centers[j]:
                    histogram[j] +=resp_weight

                        
    return histogram
    

def get_mag_grad_indices(image):
    """ get gradient and magnitude and post process them"""
    magnitude, gradient = create_grad_array(image)

    ## 2. Normalize the magnitude and round off
    mag_normed = normalize_magnitude(magnitude)
    ## 3. Absolute values of angles
    gradient = abs(gradient)
   
    return mag_normed, gradient



def get_l2_norm(array):
    """Function to return l2norm of vector"""
    array = array.flatten()
    l2_norm = np.sqrt(sum(np.square(array)))
    return l2_norm



def create_hog_features(grad_array,mag_array):
    """Function to take param gradient and magnitude and return hog features    """
    max_h = int(((grad_array.shape[0]-CELL_SIZE)/STEP_SIZE)+1)
    max_w = int(((grad_array.shape[1]-CELL_SIZE)/STEP_SIZE)+1)
    cell_array = []  ## vector to store histogram of each cell
    w = 0
    h = 0
    i = 0
    j = 0

    ### APPLY SLIDING WINDOW OF CELLS
    #Creating 8X8 cells
    while i<max_h:
        w = 0
        j = 0

        while j<max_w:
            for_hist = grad_array[h:h+CELL_SIZE,w:w+CELL_SIZE]
            for_wght = mag_array[h:h+CELL_SIZE,w:w+CELL_SIZE]
            grad_angle_cell = for_hist.flatten()
            mag_cell = for_wght.flatten()
            ## Get quantized angles
            quantized_angles = [quantize_unsigned(angle) for angle in grad_angle_cell]
                        
            ## calculate histogram of each cell
            val = get_cell_histogram(quantized_angles, mag_cell, grad_angle_cell)
            
            cell_array.append(val)
            j += 1
            w += STEP_SIZE

        i += 1
        h += STEP_SIZE

    cell_array = np.reshape(cell_array,(max_h, max_w, 9))
    #normalising blocks of cells
    block = [2,2]
    
    ### After creating cell histogram array, normalize it over the blocks

    max_h = int((max_h-block[0])+1)
    max_w = int((max_w-block[1])+1)
    block_list = []
    w = 0
    h = 0
    i = 0
    j = 0

    while i<max_h:
        w = 0
        j = 0

        while j<max_w:
            ## Take a block and normalize 
            for_norm = cell_array[h:h+block[0],w:w+block[1]]
            
            l2_norm = get_l2_norm(for_norm)
            arr_list = (for_norm/l2_norm).flatten().tolist()
            block_list+= arr_list
            j += 1
            w += 1

        i += 1
        h += 1

    
    return block_list


def normalize_hog_feature(array):
    """Function to normalize hog features"""
    sum = np.sum(array)
    ## Divide by the sum
    normalized_array = array / sum
    return normalized_array 



def get_hog_feature_of_img(imgPath):
    """ Function to return normalized hog featuer of image"""
    ## Read image
    img = cv2.imread(imgPath)
    
    ## Calculate magnitude and gradient of image
    mag_normed, gradient = get_mag_grad_indices(img)

    ## Save gradient image 
    imgFileName = os.path.basename(imgPath)
    save_path = os.path.join(path_grad_dir, imgFileName)
    cv2.imwrite(save_path, gradient)
       

    hog_features = create_hog_features(gradient, mag_normed)
    noramlized_hog_feature = normalize_hog_feature(hog_features)

    
    ## Save hog feature to .txt file

    fileName_wo_ext = imgFileName.split(".")[0]
    file_txt = fileName_wo_ext + ".txt"
    feature_file_path = os.path.join(path_feature_dir, file_txt)
    np.savetxt(feature_file_path, noramlized_hog_feature)

    return noramlized_hog_feature



def hellinger_distance(hist1, hist2):
    """Function to return hellinger distance of two vectors"""
    # Calculate the square root of the element-wise product of hist1 and hist2
    sqrt_ = np.sqrt(np.multiply(hist1, hist2))
    # Calculate the Hellinger distance    
    distance = 1 - sum(sqrt_)
    return distance

def histogram_intersection(hist1, hist2):
    """Function to return histogram intersection distance of two vectors"""
    minima = np.minimum(hist1, hist2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist2))
    return intersection


def classify_3NN(test_vector, training_vectors, training_labels, distance_metric):
    """Function to classify test vector using 3NN classifier"""
    distances = []
    for training_vector in training_vectors:
        if distance_metric==distanceMetrices[0]:
            ## distance metric Hellinger
            distance = hellinger_distance(test_vector, training_vector)
        elif distance_metric==distanceMetrices[1]:
            ## distance metric histogram intersection
            distance = histogram_intersection(test_vector, training_vector)
        distances.append(distance)
                
    sorted_distances = sorted(distances)
    top_3_distances = sorted_distances[:3]
     
    distances = np.array(distances)
    sorted_indices = np.argsort(distances)  # Get indices that would sort the distances
    nearest_indices = sorted_indices[:3]  # Select the indices of the 3 nearest neighbors
    nearest_labels = [training_labels[index] for index in nearest_indices]  # Get labels of the nearest neighbors
    predicted_label = max(set(nearest_labels), key=nearest_labels.count)  # Predict the label based on majority voting
    return predicted_label, nearest_indices, top_3_distances, nearest_labels




### Create required directories
os.makedirs(path_feature_dir, exist_ok=True)
os.makedirs(path_grad_dir, exist_ok=True)


## Get training hog feature vector


pos_train_dir = r"Image Dataset\Image Dataset\Database images (Pos)" 
neg_train_dir = r"Image Dataset\Image Dataset\Database images (Neg)"

imgs_pos = os.listdir(pos_train_dir)
imgs_neg = os.listdir(neg_train_dir)
img_paths_pos = [os.path.join(pos_train_dir, file_) for file_ in imgs_pos]
img_paths_neg = [os.path.join(neg_train_dir, file_) for file_ in imgs_neg]


train_set = []
train_labels = []
images_names = []
for imgPath in img_paths_pos:    
    label = "human"
    hog_feature =  get_hog_feature_of_img(imgPath)
    train_set.append(hog_feature)
    train_labels.append(label)
    images_names.append(os.path.basename(imgPath))
    
for imgPath in img_paths_neg:    
    label = "no_human"
    hog_feature =  get_hog_feature_of_img(imgPath)
    train_set.append(hog_feature)
    train_labels.append(label)
    images_names.append(os.path.basename(imgPath))

print(len(train_labels))
print(len(train_set))

train_array = np.array(train_set)
label_array = np.array(train_labels)

result_csv_path = os.path.join(output_dir, 'results.csv')
file_csv = open(result_csv_path, 'a+', newline="")
csvWriter = csv.writer(file_csv)

header = ['Test input image', 'Correct Classification', 'Distance Metric used' 
          'filename of 1st NN', 'distance of 1st NN', 'classification of 1st NN',
          'filename of 1st NN', 'distance of 2nd NN', 'classification of 2nd NN',
          'filename of 1st NN', 'distance of 3rd NN', 'classification of 3rd NN',
          'Classification from 3 NN', ]

csvWriter.writerow(header)



def classify_single_img(imgPath, actual_classification_label, distance_metric, trained_img_names):
    """Function to classify single test image"""
    test_vector = get_hog_feature_of_img(imgPath)

    result_label, nearest_indices, top_3_distances, nearest_labels = classify_3NN(test_vector, train_array, label_array, distance_metric)
    testImgName = os.path.basename(imgPath)
    
    fileNames3NN = [trained_img_names[index] for index in nearest_indices]
    

    ## Save result analysis to .csv file

    data = [testImgName, actual_classification_label, distance_metric,
            fileNames3NN[0], top_3_distances[0], nearest_labels[0],
            fileNames3NN[1], top_3_distances[1], nearest_labels[1],
            fileNames3NN[2], top_3_distances[2], nearest_labels[2],
            result_label
            ]
    csvWriter.writerow(data)
    
    print(f"Result {testImgName} - {result_label}")
    return result_label


## Load all test images and classify

test_images_neg_dir = r"Image Dataset\Image Dataset\Test images (Neg)"
test_images_pos_dir = r"Image Dataset\Image Dataset\Test images (Pos)"

test_pos_imgs = os.listdir(test_images_pos_dir)
test_neg_imgs = os.listdir(test_images_neg_dir)
test_pos_imgs_paths = [os.path.join(test_images_pos_dir, file_) for file_ in test_pos_imgs] 
test_neg_imgs_paths = [os.path.join(test_images_neg_dir, file_) for file_ in test_neg_imgs] 


## Classify using both the distance metrices
for key, distanceMetric in distanceMetrices.items():

    for imgPath in test_pos_imgs_paths:
        actual_label = "Human"                
        classification_result = classify_single_img(imgPath, actual_label,distanceMetric, images_names)
    for imgPath in test_neg_imgs_paths:
        actual_label = "No_human"                
        classification_result = classify_single_img(imgPath, actual_label,distanceMetric, images_names)
    
            






