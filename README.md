# prerequiste setup
- run the following line in the anaconda command prompt (shell) and make sure that environment.yml is in the same directory
  - conda env create -f environment.yml



# Meetings (what we agreed upon)
- Meeting 1
  - understand and look for papers about optical music recognition.
  - understand how to do written letters recognition.
  - Preprocesing 
    - fix image rotation using ( prespectve and affine transformation) 
    - detect angle of rotation of the image using hough transform of the lines
    - remove noise and unobvious lines
  - processing
    - detect the 11 types of the characters and how to feature-engineer them.
    
# papers 
  - papers that describe general character recognition
    - https://www.irjet.net/archives/V5/i3/IRJET-V5I3218.pdf
    

# music-notes-reader
this is the repo for the image processing project where we turn images of musical notes

# videos
- a video explaining how to read musical notes.
  - https://www.youtube.com/watch?v=Zfky3pQEeqg&ab_channel=OddQuartet
- ocr explanation computerphile
  - https://www.youtube.com/watch?v=ZNrteLp_SvY&ab_channel=Computerphile
# courses
- a course about introduction to computer science
  - https://www.udacity.com/course/introduction-to-computer-vision--ud810
- how to read a note khan academy -> very important
  - https://www.youtube.com/watch?v=wQHcz7U01M4&list=PLSQl0a2vh4HDFvmGd8eL5PJusJqrNZ0ge&ab_channel=KhanAcademyPartners
- how to deal with music in python (arabic) -> very important
  - https://www.youtube.com/watch?v=SQot7w-g7aQ&list=PLYW0LRZ3ePo7ZCXH2VFAVlTZ_b6LJeOPB&index=72&ab_channel=HussamHourani
# plan
  - update 13/12/2020
    - we need to do the following     
    
       - we need 2 people to work on extracting features of each symbol.
       
       - we need someone to integrate the work of the different stages.
       
       - we need someone to work on adjusting the image eg.music2.JPG to be similar to music1.JPG Hint(he should look for non-affine transformation)
       - someone should be responsible for the autograder https://docs.google.com/document/d/1ZxLF2Qz16wyCT-2xIju3n2G-4kNtho7g8rtM1_H7xZQ/edit
