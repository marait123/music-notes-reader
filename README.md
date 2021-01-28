   

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
2. then the horizontal lines are removed from the inverted image using morphological operations and contour finding.
3. after the horizonatal lines are removed we 
![enter image description here](https://github.com/marait123/music-notes-reader/blob/main/delivery/a_1.jpg?raw=true)

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
