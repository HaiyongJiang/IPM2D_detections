## Traditional object detections 
Programs for template matching and generalized hough transformation for a IPM project.o

Both programs highly use OpenCV libraries. As they are not optimized, the performance may be slow. However, they can be easily implemented in a parallel way by using either GPU threading or CPU threading.

Generalized hough transformation requires a proper configure of the compiling options. I provied a linux executable in the repo. 

## Notice
The original generalized hough transformation with rotation and scaling (GeneralizedHoughGuil) is implemented in a wrong way, so we utilize GeneralizedHoughBallard and manually rotate and scale the object to detect objects with both transformations.



