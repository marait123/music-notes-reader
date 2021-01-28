#Imports
from commonfunctions import *
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import skimage
from skimage.color import rgb2gray
from skimage.morphology import disk, square, rectangle as rect
from skimage.filters import threshold_otsu, threshold_local

from skimage.morphology import(binary_erosion, 
                               binary_dilation,
                               binary_closing,
                               skeletonize,
                               thin,
                               opening
                              )
from skimage.measure import find_contours

from skimage.draw import rectangle
import cv2
import scipy.stats
import pickle
import argparse
import os

##############################
# classes
class staff_meta(object):
    division_height = -1
    line_thickness = -1
    staff_lines_y_coords = []
notes_pitch = [
    'e',    # bottom line (or the 5th from above)
    'f',
    'g',
    'a',
    'b',
    'c',
    'd'
]
class notes_pitch(object):
    char =   ['e','f','g','a','b','c','d','e','f']
    factor = [ 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 2 ]

def get_trace_meta(image):
    if len(image.shape) != 2:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray = image        
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    width = gray.shape[1]//5
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width,1))

    # thresh = cv2.Canny(thresh, 30, 200) 
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    division_height=0
    h_max = 1
    bounding_rects = []
    for c in cnts:
        bounding_rects.append(cv2.boundingRect(c))

    for b_rect in bounding_rects:
        X,Y,w,h = b_rect
        if h > h_max:
            h_max = h
    def contour_y(c):
        return c[1]
    bounding_rects = sorted(bounding_rects, key=contour_y)
    lines_y_coordinates = [ b[1] for b in bounding_rects]
    div_height = bounding_rects[1][1] - bounding_rects[0][1]
    s_meta = staff_meta()
    s_meta.division_height = div_height
    s_meta.line_thickness = h_max
    s_meta.staff_lines_y_coords = lines_y_coordinates
    return s_meta

accidentals = {
    "c":"#", # this for sharp #
    "d":"", # this for the natural or neutral
    "e":"&", # this is for the b
    "ee":"&&", # this is for the bb
    "x":"##" , # this for the x or double sharp
}

numbers = {
    "0":"0",
    "1":"1",
    "2":"2",
    "3":"3",
    "4":"4",
    "5":"5",
    "6":"6",
    "7":"7",
    "8":"8",
    "9":"9"
}

duration_map = {
    "a_1":"1",
    "a_2":"2",
    "a_4":"4",
    "a_8":"8",
    "a_16":"16",
    "a_32":"32",
    "b_8":"8",
    "b_16":"16",
    "b_32":"32"
}
    
def is_accidental(char):
    return char in accidentals

def is_number(char):
    return char in numbers

accidentalFound = ['',False]
Number1_Number2 = [ ['',False],['',False] ]

def numberCheck():
    if Number1_Number2[0][1] and Number1_Number2[1][1] :
        file_object.write(f'\meter<"{Number1_Number2[0][0]}/{Number1_Number2[1][0]}"> ')
        Number1_Number2[1][1] = Number1_Number2[0][1] = False

    # Increase Accuracy as sometimes one of the numbers get detected as noise
    elif Number1_Number2[0][1]:
        file_object.write(f'\meter<"{Number1_Number2[0][0]}/{Number1_Number2[0][0]}"> ')
        Number1_Number2[1][1] = Number1_Number2[0][1] = False

    elif Number1_Number2[1][1]:
        file_object.write(f'\meter<"{Number1_Number2[1][0]}/{Number1_Number2[1][0]}"> ')
        Number1_Number2[1][1] = Number1_Number2[0][1] = False
        
def accidental_Saving(level,n):
    if accidentalFound[1]:
        file_object.write(f'{level}{ accidentals[accidentalFound[0]] }{n}/{ duration_map[char_type[0]] } ')
        accidentalFound[1] = False
        return True
    else: return False

##############################
# Functions
target_img_size = (32, 32)
def extract_hog_features(img):
    """
   (i.e win_size, cell_size, ... etc)
    """
    img = cv2.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)
    
    block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()

def get_model(fileName):
    with open(fileName, 'rb') as file:
        model = pickle.load(file)
        return model
    return None

def makeSquare(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h1 = image.shape[0]
    w1 = image.shape[1]

    w2 = max(h1, w1)
    out_image = np.zeros((w2,w2))

    if(w1 > h1):
        diff = w1 - h1
        out_image[diff//2:diff//2+h1,:] = image
    elif (w1 < h1):
        diff = h1 - w1
        out_image[:,diff//2:diff//2+w1] = image
    else:
        return image
    return out_image

def resize_to_size(image, dsize):
    return cv2.resize(image, dsize)

def pre_process(image,dsize):
    sq_image = makeSquare(image)
    resized_image = resize_to_size(sq_image, dsize)
    img = resized_image.copy()
    img = img.astype(np.uint8)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2

def get_features(img):
    img = pre_process(img, (50,50))
    features = extract_hog_features(img)  
    return features

def padd_image(img):
    out_img = np.zeros((img.shape[0]+2,img.shape[1]+2), dtype=np.uint8)
    out_img[1:out_img.shape[0]-1,1:out_img.shape[1]-1] = img
    return out_img

def get_noteLine_depending_height(centers,boxes_start_point,trace_info):
    level = []
    for center in centers:
        x = center[0]
        y = center[1]
        x += boxes_start_point[0]
        y += boxes_start_point[1]

        factor = 0
        while trace_info.staff_lines_y_coords[0] > y:
            factor += 1
            y += 3.5*(trace_info.division_height + trace_info.line_thickness)
        while trace_info.staff_lines_y_coords[4] < y:
            factor -= 1
            y -= 3.5*(trace_info.division_height + trace_info.line_thickness)
        map = []
        map.append(trace_info.staff_lines_y_coords[4])
        i = 4
        while i > 0:
            map.append((trace_info.staff_lines_y_coords[i] + trace_info.staff_lines_y_coords[i-1]) / 2)
            map.append(trace_info.staff_lines_y_coords[i-1])
            i -= 1
        map = [abs(x - y) for x in map]
        pos = map.index(min(map))
        obj = notes_pitch()
        level.append([obj.char[pos], obj.factor[pos] + factor])
    return level

def filled_holes_centers(char, radius):
    padded_char = padd_image(char)    
    erroded_image = cv2.erode(padded_char,kernel=disk(radius),iterations = 1)        
    b_boxes = get_external_bounding_boxes(erroded_image)
    centers = []
    for box in b_boxes:
        X,Y,w,h = box
        cx = X+w//2
        cy=Y+h//2
        centers.append((cx,cy))
    return centers

def halfsCenter(char, radius):
    
    padded_char = padd_image(char)
    dim = round(np.sqrt(char.shape[1]))
    se = np.ones((dim,dim))
    dilated_char = binary_dilation(padded_char,se)
    dilated_char = dilated_char.astype(np.uint8)
    dilated_char = binary_fill_holes(dilated_char).astype(np.uint8)
    erroded_image = cv2.erode(dilated_char,kernel=disk(radius),iterations = 1)
    b_boxes = get_external_bounding_boxes(erroded_image)
    centers = []
    for box in b_boxes:
        X,Y,w,h = box
        cx = X+w//2
        cy=Y+h//2
        centers.append((cx,cy))

    return centers

def WholesCenter(char):
    center = []
    x,y = char.shape
    x = x/2
    y = y/2
    center.append((x,y))
    return center

def conv_255_true(image):
    image[image == 255] = True
    image[image == 0] = False
    return image

def remove_horizontal_wrapper(org_image, inv_image):
    # both org_image and inv_image must uint8 
    line_width, no_staf_1 = remove_horizontal_lines(org_image)
    no_staf_1 = conv_255_true(no_staf_1)
    no_staf = no_staf_1
    inv_image = conv_255_true(inv_image)
    no_staf = np.bitwise_and(no_staf, inv_image)
    return no_staf

def remove_horizontal_lines(image):
    if len(image.shape) != 2:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    width = gray.shape[1]//5
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width,1))

    # thresh = cv2.Canny(thresh, 30, 200) 
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    h_max = 1
    for c in cnts:
        X,Y,w,h = cv2.boundingRect(c)
        if h > h_max:
            h_max=h
    for c in cnts:   
        cv2.drawContours(gray, [c], -1, 255, h_max)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,h_max*3))
    result = 255-cv2.morphologyEx(255 - gray, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return h_max,result

def get_external_bounding_boxes(img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bounding_rects = []
    for c in cnts:
        b_r = cv2.boundingRect(c)
        bounding_rects.append(b_r)    
    def contour_x(c):
        return c[0]
    bounding_rects = sorted(bounding_rects, key=contour_x)
    return bounding_rects

def draw_external_bounding_boxes(img,b_rects):
    out = img.copy()
    for b_r in b_rects:
        X,Y,w,h = b_r
        out = cv2.rectangle(out, (X,Y), (X+w, Y+h),1,3)
    return out

def binary_opening(img, selem):
    return binary_dilation(binary_erosion(img, selem=selem), selem=selem)

def get_bounding_boxes(img,box_aspect_ratio):
    contours = find_contours(img,.8, fully_connected="low")
    bounding_boxes=[]
    for contour in contours:
        Xmin= np.min(contour[:,1])
        Xmax=np.max(contour[:,1]) 
        Ymin=np.min(contour[:,0])
        Ymax = np.max(contour[:,0])
        if(Ymax-Ymin != 0):
            aspect_ratio = (Xmax-Xmin)/(Ymax-Ymin)
        else:
            aspect_ratio = 3000
            
        if aspect_ratio >= box_aspect_ratio[0] and aspect_ratio <= box_aspect_ratio[1]:
            bounding_boxes.append([int(round(Xmin)),int(round(Xmax)),int(round(Ymin)),int(round(Ymax))])

    def x_min(shape):
        return shape[0]             

    bounding_boxes = sorted(bounding_boxes, key=x_min)
    return bounding_boxes


def get_bounded_image(img,bounding_boxes):
    conte = np.zeros(img.shape)
    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        rr, cc = rectangle(start = (Ymin,Xmin), end = (Ymax,Xmax), shape=img.shape)
        rr = rr.astype(np.int)
        cc= cc.astype(np.int)
        conte[rr, cc] = 1 #set color white
    img2 = np.where(conte==1, img, 0)
    return img2

#####################################
# load the models
training_files_names = [
    '0_oval_model_4',
    '1_oval_model_4',
    '2_oval_model_4',
]

models = []
for train_file in training_files_names:
    model = get_model(f'models/{train_file}')
    models.append(model)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("inputfolder", help = "Input File")
parser.add_argument("outputfolder", help = "Output File")
args = parser.parse_args()

testCases = sorted(os.listdir(f"{args.inputfolder}"))

TestNumber = 1
for testImg in testCases:
    original_image = cv2.imread(f"{args.inputfolder}/{testImg}" )
    gray_image = np.round(rgb2gray(original_image)*255)
    binary_image = binarize(gray_image)
    inverted_image = 255 - binary_image

    # img be binarized image of size height x width and foregound pixel is white i.e pixel value = 255
    horizontal_hist = np.sum(inverted_image,axis=1,keepdims=True)/255
    peaks = 0
    offset = 1
    j = 0
    th = int(round(np.max(horizontal_hist)*.95))
    staff_lines_y=[]
    while j < len(horizontal_hist):
        while j < len(horizontal_hist) and horizontal_hist[j] < th:
            j+=1
        if j >= len(horizontal_hist):
            break
        staff_lines_y.append(j)    
        peaks+=1
        st = j
        while j < len(horizontal_hist) and horizontal_hist[j] >= th:
            j+=offset
    number_of_staffs = int(round(peaks/ 5))
    min_areas_coords = []
    min_areas_coords.append([0, staff_lines_y[0]])

    j = 4
    while j < len(staff_lines_y) - 5:
        min_areas_coords.append([staff_lines_y[j],staff_lines_y[j+1]])
        j+=5
    min_areas_coords.append([staff_lines_y[len(staff_lines_y)-1],gray_image.shape[0]])
    division_height = staff_lines_y[1] - staff_lines_y[0]
    thresh_span = division_height # minimum must span thin number of pixels
    separation_ys = []

    for min_pair in min_areas_coords:
        i = min_pair[0]
        minimum = i
        min_span = 1
        i+=1
        while i < min_pair[1]:
            if horizontal_hist[minimum] <= horizontal_hist[i]:
                min_span += 1
            else:
                min_span=1
                minimum = i
            i+=1
        separation_ys.append(minimum)

    staff_boxes = [] # stored as Xmin, Ymin, W, H
    for i in range(len(separation_ys)-1):
        W =inverted_image.shape[1]
        H = separation_ys[i+1] - separation_ys[i]
        staff_boxes.append([0, separation_ys[i], W, H])

    file_object = open(f"{args.outputfolder}/{TestNumber}.txt","w+")

    if len(staff_boxes) > 1:
        file_object.write('{\n')

    for i,box in enumerate(staff_boxes):
        file_object.write('[ ')
        Xmin,Ymin, W, H = box
        Ymax = Ymin+H
        Xmax = Xmin + W
        orig_inv_staff = inverted_image[Ymin:Ymax,Xmin:Xmax].astype(np.uint8)
        no_staf=remove_horizontal_wrapper(original_image[Ymin:Ymax,Xmin:Xmax],inverted_image[Ymin:Ymax,Xmin:Xmax].astype(np.uint8) )
        char_boxes = get_external_bounding_boxes(no_staf)
        out = draw_external_bounding_boxes(no_staf, char_boxes)
        koko = 0
        trace_info: staff_meta = get_trace_meta(original_image[Ymin:Ymax,Xmin:Xmax])
        for box in char_boxes:
            [Xmin, Ymin, W, H] = [x for x in box]
            # remove blank or noisy area that is less than a line of pixels
            char = no_staf[Ymin:Ymin+H,Xmin:Xmin+W]
            # remove it
            radius = int(trace_info.division_height // 3 )
            oval_centers = filled_holes_centers(char, radius)

            if W < trace_info.division_height//2 and H < trace_info.division_height//2:
                # use the properties of a dot or circle to know for sure it is a dot
                erroded = padd_image(char)
                aspect_ratio = W/H
                if(aspect_ratio > .5 and aspect_ratio < 1.5 ):
                    erroded = cv2.morphologyEx(erroded, cv2.MORPH_ERODE, disk(W//4), iterations=1)
            else:
                if len(oval_centers) == 0:
                    char_type = models[0].predict([get_features(char)])

                    if char_type[0] != 'noise':
                        if is_accidental( char_type[0] ) :
                            accidentalFound = [ char_type[0] , True ]
                        elif is_number( char_type[0] ) :
                            if Number1_Number2[0][1]:
                                Number1_Number2[1][1] = True
                                Number1_Number2[1][0] = char_type[0]
                            else:
                                Number1_Number2[0][1] = True
                                Number1_Number2[1][0] = char_type[0]
                        # Decide to print meter or not
                        else:
                            # May Miss one or both of the numbers as noise
                            if char_type[0] != 'z': 
                                numberCheck()

                                if char_type[0] == 'a_1':
                                    oval_centers = WholesCenter(char)
                                    level = get_noteLine_depending_height(oval_centers,(Xmin,Ymin),trace_info)
                    
                                elif char_type[0] == 'a_2':
                                    oval_centers = halfsCenter(char,radius)
                                    level = get_noteLine_depending_height(oval_centers,(Xmin,Ymin),trace_info)
                                
                                if len(level) > 0 and accidental_Saving(level[0][0],level[0][1]) == False:
                                    file_object.write(f'{level[0][0]}{level[0][1]}/{ duration_map[char_type[0]] } ')
                                
                elif len(oval_centers) == 1:
                    char_type = models[1].predict([get_features(char)])
                    if char_type[0] != 'z':
                        numberCheck()
                        level = get_noteLine_depending_height(oval_centers,(Xmin,Ymin),trace_info)
                        if accidental_Saving(level[0][0],level[0][1]) == False:
                            file_object.write(f'{level[0][0]}{level[0][1]}/{duration_map[char_type[0]]} ')
                elif len(oval_centers) >= 2:
                    # using the x difference between the 1st 2 centre we can separate between chords 
                    # or beams
                    numberCheck()
                    if oval_centers[1][0] - oval_centers[0][0] > int(1.5*trace_info.division_height):
                        # the model has very low accuracy

                        beam_type = models[2].predict([get_features(char)])
                        levels = get_noteLine_depending_height(oval_centers,(Xmin,Ymin),trace_info)
                        for level in levels:
                            if accidentalFound[1] and level == levels[0] :
                                file_object.write(f'{level[0]}{ accidentals[accidentalFound[0]] }{level[1]}/{ duration_map[beam_type[0]] } ')
                                accidentalFound[1] = False
                            else:
                                file_object.write(f"{level[0]}{level[1]}/{ duration_map[beam_type[0]] } ")
                    else:
                        chord_type = models[2].predict([get_features(char)])
                        levels = get_noteLine_depending_height(oval_centers,(Xmin,Ymin),trace_info)
                        file_object.write('{')
                        for level in levels:
                            file_object.write(f"{level[0]}{level[1]}/{duration_map[chord_type[0]]}")
                            if level != levels[len(levels)-1]:
                                file_object.write(',')
                        file_object.write('} ')
            
        file_object.write(']\n')
    
    if len(staff_boxes) > 1:
        file_object.write('}')
    file_object.close()
    TestNumber += 1  
