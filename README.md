# GMH-D
A simple method based on RGB-D camera and Google MediaPipe for accurate 3D hand tracking

-------------------------------

This repository contains the script for running Google MediaPipe (GMH) and its enhanced version (GMH-D), a 3D handtracking method for tracking accurately the hand and its fingers in clinical videos. The method has been validated with respect to motion capture for the characterization of the Finger Tapping task, the Hand Opening Closing task, and the Multiple Fingers Tapping task, but can be extended and adapted to other clinical tastk. The method runs at 30 fps on a normal laptop and can process both real-streams of data using opencv libraries or offline on MKV files recorded by Azure Kinect.

To know more about the method, you can refer to:

- G. Amprimo, C. Ferraris, G. Masi, G. Pettiti and L. Priano, "GMH-D: Combining Google MediaPipe and RGB-Depth Cameras for Hand Motor Skills Remote Assessment," 2022 IEEE International Conference on Digital Health (ICDH), Barcelona, Spain, 2022, pp. 132-141, doi: 10.1109/ICDH55609.2022.00029.
   

The main script is main.py. The file requirements.txt contains the dependencies to run the code. The code has currently been implemented for Azure Kinect only and relies on Azure libraries and their porting to Python for working. We are planning to implement the method also for other RGB-D cameras such as the Intel Real Sense and Smartphones camera embedding a depth stream such as Iphone and Ipad. If you want to contribute, feel free to collaborate :-)

If you want to reuse the code for your work, please cite at least one of the papers reported above.

