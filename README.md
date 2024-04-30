**Gesture-Controlled Mouse with MediaPipe and PyAutoGUI Custom Library**

## Introduction
This project offers a solution for controlling mouse movement using hand gestures recognized by MediaPipe's AI. It enables users to manipulate the mouse cursor with one hand while utilizing the other hand for mouse clicks. The PyAutoGUI library has been customized to enhance performance on macOS, incorporating compatibility with Cython for optimized execution.

## Features
- **Gesture Recognition**: Utilizes MediaPipe to detect hand gestures and nodes collision for precise control.
- **Custom PyAutoGUI Library**: Improved PyAutoGUI library with macOS performance enhancements and Cython compatibility.
- **Cython Implementation**: Developed with Cython for enhanced performance and smoother operation, achieving a stable 30 frames per second (FPS).
- **Facial Recognition**: Implements facial recognition to ensure only the user captured by the camera can control their PC with hand gestures. Users need to provide their photo in the designated folder within the project for facial recognition.

## Installation
1. Ensure Python and the Cython compiler are installed on your system.
2. Clone the repository:
   ```
   git clone https://github.com/eliorodr2104/Mouse_AR.git
   ```

## Usage
1. Ensure the camera is properly connected and accessible.
2. Run the main script:
   ```
   python setup.py
   ```
4. Optionally, place your photo in the designated folder for facial recognition.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
- MediaPipe: [Link](https://github.com/google/mediapipe)
- PyAutoGUI: [Link](https://github.com/asweigart/pyautogui)
