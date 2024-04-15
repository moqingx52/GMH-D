# GMH-D
A simple method based on RGB-D camera and Google MediaPipe for accurate 3D hand tracking

-------------------------------

This repository contains the script for running Google Mediapipe Hands-Depth (GMH-D), a depth-enhanced version of Google Mediapipe Hands for 3D tracking the hand and its fingers, for several applications, including automatic clinical evaluation of hand movements in Parkinson's disease from video. GMH-D surpasses normal Google Mediapipe Hands (GMH) in the estimation of spatial motion, as proved in our research work. Indeed, the method has been validated with respect to motion capture for the characterization of the Finger Tapping task, the Hand Opening Closing task, and the Multiple Fingers Tapping task, but can be extended and adapted for tracking other type of fine finger movements. The method runs at 30 fps on  Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz 2.40 GHz with 16 GB RAM and can process both real-time streams of data or offline MKV recordings by Kinect Azure. 

To know more about the method, you can refer to:

```bash
- G. Amprimo, C. Ferraris, G. Masi, G. Pettiti and L. Priano,
"GMH-D: Combining Google MediaPipe and RGB-Depth Cameras for Hand Motor Skills Remote Assessment,"
2022 IEEE International Conference on Digital Health (ICDH), Barcelona, Spain, 2022, pp. 132-141,
doi: 10.1109/ICDH55609.2022.00029.
```

```bash
- G. Amprimo, G. Masi, G. Pettiti, G. Olmo, L. Priano, C. Ferraris
"Hand tracking for clinical applications: validation of the google mediapipe hand (GMH) and
 the depth-enhanced GMH-D frameworks."
arXiv preprint arXiv:2308.01088 (2023).
```

If you plan to use this code, please consider citing the above publications in your work. 

While the method was developed and validated for Azure Kinect, it can be extended to any RGB-D camera. Currently, Intel RealSense D4XX family of cameras is supported and the code for Azure Kinect may be easily adapted for Orbecc FEMTO cameras, using a wrapper of Azure Kinect SDK. Future plans for porting the solution on smartphones and tablets embedding RGB-D cameras (Ipad, IPhone) have been planned. If you want to contribute, feel free to collaborate to the project :-)

----------

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/<username>/GMHD_Hand_Tracking.git
    ```
3. Create a Python environment (optional but recommended):

    ```bash
    python -m venv env
    ```

4. Activate the Python environment:

    - **Windows:**

    ```bash
    .\env\Scripts\activate
    ```

    - **Linux/macOS:**

    ```bash
    source env/bin/activate
    ```

5. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Code - AZURE KINECT (TESTED)

The main entry point of the code is the `GMHD_AzureKinect.py` script. You can run the script with various options to specify the processing mode, input file paths, and other parameters:

```bash
python GMHD_AzureKinect.py --mode <mode> --mkvfilepath <mkvfilepath> --mkvfilename <mkvfilename> --save <yes/no> --outputpath <outputpath> --outputname <outputname> --n_hands <n_hands> --handconf <handconf> --rerun_pd <rerun_pd> --jointconf <jointconf> --interval <interval> --visualize <yes/no>
```

#### Options:

- `mode`: Processing mode (`offline` or `online`).
- `mkvfilepath`: Path to the folder containing the MKV file.
- `mkvfilename`: MKV file name.
- `save`: Save tracked joints (`yes` or `no`).
- `outputpath`: Absolute path to the output folder (only if `save` is `yes`).
- `outputname`: Name for the tracking output JSON file (only if `save` is `yes`).
- `n_hands`: Number of hands to track (should be `>=1`).
- `handconf`: Confidence threshold for hand tracking (range: `[0, 1]`).
- `rerun_pd`: Confidence of detection before rerunning Palm Detector (range: `[0, 1]`).
- `jointconf`: Confidence threshold for joint tracking (range: `[0, 1]`).
- `interval`: Set greater than `0` for automatically recording `t` seconds (range: `[1, +inf]`).
- `visualize`: Visualize tracking while processing video (`yes` or `no`).

### Examples:

#### Offline Mode:

```bash
python GMHD_AzureKinect.py --mode offline --mkvfilepath /path/to/mkv/folder --mkvfilename example.mkv --save yes --outputpath /path/to/output/folder --outputname tracking_data --n_hands 1 --handconf 0.5 --rerun_pd 0.2 --jointconf 0.5 --visualize yes
```

#### Online Mode:

```bash
python GMHD_AzureKinect.py --mode online --save no --n_hands 1 --handconf 0.5 --rerun_pd 0.2 --jointconf 0.5 --interval 5 --visualize yes
```






### Running the Code - Intel RealSense D4xx cameras (TESTED with Intel RealSense D415)

The main entry point of the code is the `GMHD_RealSense.py` script. You can run the script with various options to specify the processing mode, input file paths, and other parameters:

```bash
python GMHD_AzureKinect.py --mode <mode> --bagfilepath <mkvfilepath> --bagfilename <mkvfilename> --save <yes/no> --outputpath <outputpath> --outputname <outputname> --n_hands <n_hands> --handconf <handconf> --rerun_pd <rerun_pd> --jointconf <jointconf> --interval <interval> --visualize <yes/no>
```

#### Options:

- `mode`: Processing mode (`offline` or `online`).
- `bagfilepath`: Path to the folder containing the pre-recorded bag file.
- `bagfilename`: bag file name.
- `save`: Save tracked joints (`yes` or `no`).
- `outputpath`: Absolute path to the output folder (only if `save` is `yes`).
- `outputname`: Name for the tracking output JSON file (only if `save` is `yes`).
- `n_hands`: Number of hands to track (should be `>=1`).
- `handconf`: Confidence threshold for hand tracking (range: `[0, 1]`).
- `rerun_pd`: Confidence of detection before rerunning Palm Detector (range: `[0, 1]`).
- `jointconf`: Confidence threshold for joint tracking (range: `[0, 1]`).
- `interval`: Set greater than `0` for automatically recording `t` seconds (range: `[1, +inf]`).
- `visualize`: Visualize tracking while processing video (`yes` or `no`).

### Examples:

#### Offline Mode:

```bash
python GMHD_RealSense.py --mode offline --bagfilepath /path/to/bag/folder --bagfilename example.bag --save yes --outputpath /path/to/output/folder --outputname tracking_data --n_hands 1 --handconf 0.5 --rerun_pd 0.2 --jointconf 0.5 --visualize yes
```

#### Online Mode:

```bash
python GMHD_RealSense.py --mode online --save no --n_hands 1 --handconf 0.5 --rerun_pd 0.2 --jointconf 0.5 --interval 5 --visualize yes
```

## License

This code is licensed under the [MIT License](LICENSE).


