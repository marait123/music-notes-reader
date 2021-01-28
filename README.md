   


[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
![downloads](https://img.shields.io/github/downloads/marait123/music-notes-reader/total)![repo size](https://img.shields.io/github/repo-size/marait123/music-notes-reader)

[![GitHub contributors](https://img.shields.io/github/contributors/marait123/music-notes-reader)](https://github.com/marait123/music-notes-reader/graphs/contributors)

[![GitHub forks](https://img.shields.io/github/forks/marait123/music-notes-reader)](https://github.com/marait123/music-notes-reader/network/members)
[![GitHub stars](https://img.shields.io/github/stars/marait123/music-notes-reader)](https://github.com/marait123/music-notes-reader/stargazers)


## How to Run
 1. install Conda
 2. Install [conda](https://www.anaconda.com/products/individual)
 3. run the following
```
conda env create -f requirements.yml

conda activate Image-Project

python3 main.py <input directory path> <output directory path>
e.g. python3 main.py test-cases test-out --> this will generate parse the input folder test-cases and for each image in it it will produce file that will be placed in test-out folder
```
## How It Works
### the pipeline

1. the input image is converted to binary using otsu's method then it is inverted so that the symbols of interest have higher pixel values so they're easier to play with
![enter image description here](https://github.com/marait123/music-notes-reader/blob/main/delivery/gray.png?raw=true)
![enter image description here](https://github.com/marait123/music-notes-reader/blob/main/delivery/inverted.png?raw=true)
2. then the image is segmented using histogram projection analysis of the inverted image to a number of staffs 3 in this case
![enter image description here](https://github.com/marait123/music-notes-reader/blob/main/delivery/inverted_histogram.png?raw=true)
![enter image description here](https://github.com/marait123/music-notes-reader/blob/main/delivery/box1.png?raw=true)![enter image description here](https://github.com/marait123/music-notes-reader/blob/main/delivery/box2.png?raw=true)
![enter image description here](https://github.com/marait123/music-notes-reader/blob/main/delivery/box3.png?raw=true)

3. then we loop on the number of resulting boxes from the previous step and for each box we remove the horizontal lines using morphological operations and contour finding then the damaged image symbols are reconstructed using column dialation and anding operation with the original image.
4.  using find contours we segment the resulting no_staff_line image of the staff to a number of subimages each containing one symbol.
![enter image description here](https://github.com/marait123/music-notes-reader/blob/main/delivery/symbols.JPG?raw=true)
5. the segmented images from the  previous step can be classified into 3 groups of symbols based on the  length of the array resulting from filled_holes_centers, which returns the number of ovals in the symbol image. 
6. if number of ovals in the image is 0 then it is either segmentation noise or symbols with no ovals like accidentals or numbers and we distinguish between them by using a support vector machine model  (model[0]) that is trained on a dataset that we made for ourselves
7. if number of ovals in the image is 1 then it is either quarter ,8th or 32nd note or sometimes g-clef. we disinguish between these symbols using another model(model[1]) that is trained to dintinguish between these symbols.
8. if number of ovals is >1 then the symbol is a beam or chord.
9. using the oval centres returned from filled_holes_centers we can know by comparing these to the y coordinates of the staff lines the pitch of the note.
10. note that we consider images that has length or width less that a certain threshold a dot symbol. 
# Useful Resources
this is the repo for the image processing project where we turn images of musical notes

# papers 
  - [papers that describe sgeneral character recognition](https://www.irjet.net/archives/V5/i3/IRJET-V5I3218.pdf) 

## videos
- [a video explaining how to read musical notes]( https://www.youtube.com/watch?v=Zfky3pQEeqg&ab_channel=OddQuartet).  
- [ocr explanation computerphile](https://www.youtube.com/watch?v=ZNrteLp_SvY&ab_channel=Computerphile)  
## courses
- [a course about introduction to computer vision](https://www.udacity.com/course/introduction-to-computer-vision--ud810)
- [how to read a note khan academy]( https://www.youtube.com/watch?v=wQHcz7U01M4&list=PLSQl0a2vh4HDFvmGd8eL5PJusJqrNZ0ge&ab_channel=KhanAcademyPartners) -> very important
  
- [how to deal with music in python (arabic)](https://www.youtube.com/watch?v=SQot7w-g7aQ&list=PLYW0LRZ3ePo7ZCXH2VFAVlTZ_b6LJeOPB&index=72&ab_channel=HussamHourani)-> very important

## Dataset
- [scanned dataset of the project prepared by our team](https://drive.google.com/drive/u/0/folders/1eJctuSXNN3N60AC2Ikmuk8w53-pgjysp)
