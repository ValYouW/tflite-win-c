# tflite-win-c
This is an ssd object detection and deeplab image segmentation demo project using TensorFlow Lite C API on windows with Visual Studio C++.

# Setup
## OpenCV
The project make use of OpenCV:
1. [Download](https://www.opencv.org/releases) and extract opencv to some folder, e.g `c:\tools\opencv`
1. Define global environment variable `OPENCV_DIR` pointing the opencv install dir. The project reference opencv `build\include` dir and `build\x64\vc15\lib` dirs as follow: `$(OPENCV_DIR)\build\include` and `$(OPENCV_DIR)\build\x64\vc15\lib`, so make sure `OPENCV_DIR` points to the right place.
1. During runtime, the app will need to load opencv dll, make sure to have opencv `build\x64\vc15\bin` dir in the path.

## tflite-dist
The project need TensorFlow Lite headers, C lib and C dll, either [download them from here](https://github.com/ValYouW/tflite-dist/releases) or build it yourself. Eventually there should be a "tflite-dist" as follow:
```
+- tflite-dist
+---+ include
+-------+ tenslorflow\lite\c (all c headers)
+---+ libs\windows_x86_64\tensorflowlite_c.dll.if.lib
+---+ libs\windows_x86_64\tensorflowlite_c.dll
```

The project refernce these files using a `TFLITE_DIR` environment variable:
1. Define a global environment variable that points to your `tflite-dist` folder.
1. **Make sure to select the "Release/x64" build configuration**
1. During runtime, the app will need to load `tensorflowlite_c.dll`, make sure to have `tflite-dist\libs\windows_x86_64` dir in the path.

# License
MIT
