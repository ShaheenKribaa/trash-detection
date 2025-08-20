import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_v2/tflite_v2.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        brightness: Brightness.light,
        useMaterial3: true,
      ),
      home: const ObjectDetectionScreen(),
    );
  }
}

class ObjectDetectionScreen extends StatefulWidget {
  const ObjectDetectionScreen({super.key});

  @override
  State<ObjectDetectionScreen> createState() => _ObjectDetectionScreenState();
}

class _ObjectDetectionScreenState extends State<ObjectDetectionScreen> {
  final ImagePicker _picker = ImagePicker();
  XFile? _image;
  File? file;
  dynamic _recognitions;
  String v = "";
  bool _isDetecting = false;
  bool _isFullScreen = false;

  // Camera variables
  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isCameraInitialized = false;
  List<DetectionResult> _detectionResults = [];
  Timer? _detectionTimer;
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    loadmodel().then((value) {
      setState(() {});
    });
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _detectionTimer?.cancel();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    try {
      // Request camera permission
      var status = await Permission.camera.status;
      if (!status.isGranted) {
        status = await Permission.camera.request();
        if (!status.isGranted) {
          print('Camera permission denied');
          return;
        }
      }

      _cameras = await availableCameras();
      if (_cameras != null && _cameras!.isNotEmpty) {
        _cameraController = CameraController(
          _cameras![0],
          ResolutionPreset.high,
          enableAudio: false,
        );
        await _cameraController!.initialize();
        setState(() {
          _isCameraInitialized = true;
        });
      }
    } catch (e) {
      print('Error initializing camera: $e');
    }
  }

  Future<void> loadmodel() async {
    await Tflite.loadModel(
      model: "assets/model_unquant.tflite",
      labels: "assets/labels.txt",
    );
  }

  Future<void> _pickImage() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        setState(() {
          _image = image;
          file = File(image.path);
          _isFullScreen = false;
        });
        detectimage(file!);
      }
    } catch (e) {
      print('Error picking image: $e');
    }
  }

  Future<void> _openCamera() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.camera);
      if (image != null) {
        setState(() {
          _image = image;
          file = File(image.path);
          _isFullScreen = false;
        });
        detectimage(file!);
      }
    } catch (e) {
      print('Error taking photo: $e');
    }
  }

  Future<void> _startLiveDetection() async {
    if (!_isCameraInitialized || _cameraController == null) {
      setState(() {
        v = "Camera not initialized";
      });
      return;
    }

    setState(() {
      _isDetecting = true;
      _isFullScreen = true;
      v = "Live detection started...";
    });

    // Start periodic detection with better timing
    _detectionTimer = Timer.periodic(const Duration(milliseconds: 3000), (
      timer,
    ) async {
      if (!_isDetecting || _isProcessing) {
        return;
      }

      try {
        _isProcessing = true;
        final XFile image = await _cameraController!.takePicture();
        await _detectAndUpdate(image);
        _isProcessing = false;
      } catch (e) {
        print('Error in live detection: $e');
        _isProcessing = false;
      }
    });
  }

  Future<void> _detectAndUpdate(XFile image) async {
    try {
      var recognitions = await Tflite.runModelOnImage(
        path: image.path,
        numResults: 6,
        threshold: 0.05,
        imageMean: 127.5,
        imageStd: 127.5,
      );

      if (recognitions != null && recognitions.isNotEmpty) {
        print('Raw recognition results: $recognitions');
        List<DetectionResult> results = [];

        // Check if the model provides bounding box coordinates
        bool hasBoundingBoxes =
            recognitions.isNotEmpty && recognitions[0].containsKey('rect') ||
            recognitions[0].containsKey('boundingBox') ||
            recognitions[0].containsKey('left') ||
            recognitions[0].containsKey('x');

        print('Has bounding boxes: $hasBoundingBoxes');

        if (hasBoundingBoxes) {
          // Model provides actual bounding box coordinates
          for (var recognition in recognitions) {
            if (recognition['confidence'] > 0.1) {
              double left, top, width, height;

              // Handle different bounding box formats
              if (recognition.containsKey('rect')) {
                var rect = recognition['rect'];
                left = rect['left'] ?? 0.0;
                top = rect['top'] ?? 0.0;
                width = rect['width'] ?? 0.0;
                height = rect['height'] ?? 0.0;
              } else if (recognition.containsKey('boundingBox')) {
                var bbox = recognition['boundingBox'];
                left = bbox['left'] ?? 0.0;
                top = bbox['top'] ?? 0.0;
                width = bbox['width'] ?? 0.0;
                height = bbox['height'] ?? 0.0;
              } else if (recognition.containsKey('left')) {
                left = recognition['left'] ?? 0.0;
                top = recognition['top'] ?? 0.0;
                width = recognition['width'] ?? 0.0;
                height = recognition['height'] ?? 0.0;
              } else if (recognition.containsKey('x')) {
                left = recognition['x'] ?? 0.0;
                top = recognition['y'] ?? 0.0;
                width = recognition['w'] ?? 0.0;
                height = recognition['h'] ?? 0.0;
              } else {
                // Fallback to fixed positions
                left = 0.1;
                top = 0.1;
                width = 0.3;
                height = 0.3;
              }

              results.add(
                DetectionResult(
                  label: recognition['label'],
                  confidence: recognition['confidence'],
                  boundingBox: Rect.fromLTWH(left, top, width, height),
                ),
              );
            }
          }
        } else {
          // Current model doesn't provide bounding boxes - use single centered box
          if (recognitions.isNotEmpty) {
            var recognition =
                recognitions[0]; // Use the highest confidence detection
            if (recognition['confidence'] > 0.1) {
              // Create a single centered box
              double left = 0.25; // 25% from left
              double top = 0.25; // 25% from top
              double width = 0.5; // 50% of screen width
              double height = 0.5; // 50% of screen height

              results.add(
                DetectionResult(
                  label: recognition['label'],
                  confidence: recognition['confidence'],
                  boundingBox: Rect.fromLTWH(left, top, width, height),
                ),
              );
            }
          }
        }

        setState(() {
          _detectionResults = results;
          if (results.isNotEmpty) {
            String resultText = "Detected Objects:\n";
            for (var result in results) {
              resultText +=
                  "${result.label} - ${(result.confidence * 100).toStringAsFixed(1)}%\n";
            }
            v = resultText;
            print('Detection results: $resultText');
          } else {
            v = "No objects detected";
            print('No objects detected with confidence > 0.1');
          }
        });
      }
    } catch (e) {
      print('Error detecting objects: $e');
    }
  }

  void _stopLiveDetection() {
    setState(() {
      _isDetecting = false;
      _isFullScreen = false;
      v = "Live detection stopped";
      _detectionResults.clear();
    });
    _detectionTimer?.cancel();
  }

  Future<void> detectimage(File image) async {
    var recognitions = await Tflite.runModelOnImage(
      path: image.path,
      numResults: 6,
      threshold: 0.05,
      imageMean: 127.5,
      imageStd: 127.5,
    );
    setState(() {
      _recognitions = recognitions;
      if (recognitions != null && recognitions.isNotEmpty) {
        String resultText = "Detected Objects:\n";
        for (int i = 0; i < recognitions.length; i++) {
          var recognition = recognitions[i];
          if (recognition['confidence'] > 0.1) {
            resultText +=
                "${i + 1}. ${recognition['label']} - ${(recognition['confidence'] * 100).toStringAsFixed(1)}%\n";
          }
        }
        v = resultText;
      } else {
        v = "No objects detected";
      }
    });
    print(_recognitions);
  }

  @override
  Widget build(BuildContext context) {
    if (_isFullScreen && _isDetecting && _isCameraInitialized) {
      return _buildFullScreenCamera();
    }

    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.deepPurple,
        title: const Text(
          'AI Object Detection',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 20,
          ),
        ),
        centerTitle: true,
      ),
      body: Column(
        children: [
          // Main content area
          Expanded(
            flex: 3,
            child: Container(
              margin: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.1),
                    spreadRadius: 1,
                    blurRadius: 10,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: _image != null
                    ? Stack(
                        children: [
                          Image.file(
                            File(_image!.path),
                            fit: BoxFit.cover,
                            width: double.infinity,
                            height: double.infinity,
                          ),
                          if (_detectionResults.isNotEmpty)
                            CustomPaint(
                              painter: BoundingBoxPainter(_detectionResults),
                              child: Container(),
                            ),
                        ],
                      )
                    : Container(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [
                              Colors.deepPurple.withOpacity(0.1),
                              Colors.purple.withOpacity(0.1),
                            ],
                          ),
                        ),
                        child: const Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.camera_alt_outlined,
                                size: 80,
                                color: Colors.deepPurple,
                              ),
                              SizedBox(height: 16),
                              Text(
                                'Camera Preview',
                                style: TextStyle(
                                  fontSize: 24,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.deepPurple,
                                ),
                              ),
                              SizedBox(height: 8),
                              Text(
                                'Select an image or start live detection',
                                style: TextStyle(
                                  fontSize: 16,
                                  color: Colors.grey,
                                ),
                              ),
                              SizedBox(height: 16),
                              Text(
                                'Note: Current model is classification-based.\nShows centered detection indicator.\nFor accurate object tracking, use an object detection model.',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.grey,
                                ),
                                textAlign: TextAlign.center,
                              ),
                            ],
                          ),
                        ),
                      ),
              ),
            ),
          ),

          // Controls section
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(20),
                topRight: Radius.circular(20),
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.grey.withOpacity(0.1),
                  spreadRadius: 1,
                  blurRadius: 10,
                  offset: const Offset(0, -2),
                ),
              ],
            ),
            child: Column(
              children: [
                // Action buttons
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _buildActionButton(
                      icon: Icons.photo_library,
                      label: 'Gallery',
                      onPressed: _pickImage,
                      color: Colors.blue,
                    ),
                    _buildActionButton(
                      icon: Icons.camera_alt,
                      label: 'Camera',
                      onPressed: _openCamera,
                      color: Colors.green,
                    ),
                  ],
                ),
                const SizedBox(height: 20),

                // Live detection buttons
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _buildActionButton(
                      icon: Icons.play_arrow,
                      label: 'Start Live',
                      onPressed: _isDetecting ? null : _startLiveDetection,
                      color: _isDetecting ? Colors.grey : Colors.deepPurple,
                    ),
                    _buildActionButton(
                      icon: Icons.stop,
                      label: 'Stop Live',
                      onPressed: _isDetecting ? _stopLiveDetection : null,
                      color: _isDetecting ? Colors.red : Colors.grey,
                    ),
                  ],
                ),
                const SizedBox(height: 20),

                // Results display
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.grey[100],
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.grey[300]!),
                  ),
                  child: Text(
                    v.isEmpty ? "Detection results will appear here" : v,
                    style: TextStyle(
                      fontSize: 16,
                      color: v.isEmpty ? Colors.grey : Colors.black87,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required VoidCallback? onPressed,
    required Color color,
  }) {
    return Expanded(
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 8),
        child: ElevatedButton.icon(
          onPressed: onPressed,
          icon: Icon(icon, color: Colors.white),
          label: Text(
            label,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
          style: ElevatedButton.styleFrom(
            backgroundColor: color,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            elevation: 2,
          ),
        ),
      ),
    );
  }

  Widget _buildFullScreenCamera() {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Camera preview
          SizedBox(
            width: double.infinity,
            height: double.infinity,
            child: CameraPreview(_cameraController!),
          ),

          // Bounding boxes overlay
          if (_detectionResults.isNotEmpty)
            CustomPaint(
              painter: BoundingBoxPainter(_detectionResults),
              child: Container(),
            ),

          // Top controls
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.only(
                top: MediaQuery.of(context).padding.top + 16,
                left: 16,
                right: 16,
                bottom: 16,
              ),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Colors.black.withOpacity(0.7), Colors.transparent],
                ),
              ),
              child: Row(
                children: [
                  IconButton(
                    onPressed: _stopLiveDetection,
                    icon: const Icon(
                      Icons.close,
                      color: Colors.white,
                      size: 30,
                    ),
                  ),
                  const Spacer(),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.5),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Text(
                      'Live Detection Active',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Bottom controls
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: EdgeInsets.only(
                bottom: MediaQuery.of(context).padding.bottom + 16,
                left: 16,
                right: 16,
                top: 16,
              ),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.bottomCenter,
                  end: Alignment.topCenter,
                  colors: [Colors.black.withOpacity(0.7), Colors.transparent],
                ),
              ),
              child: Column(
                children: [
                  // Detection results
                  if (v.isNotEmpty && v != "Live detection started...")
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(12),
                      margin: const EdgeInsets.only(bottom: 16),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.7),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        v,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),

                  // Control buttons
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      _buildFullScreenButton(
                        icon: Icons.stop,
                        label: 'Stop',
                        onPressed: _stopLiveDetection,
                        color: Colors.red,
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFullScreenButton({
    required IconData icon,
    required String label,
    required VoidCallback onPressed,
    required Color color,
  }) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 60,
          height: 60,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: color.withOpacity(0.3),
                spreadRadius: 2,
                blurRadius: 8,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: IconButton(
            onPressed: onPressed,
            icon: Icon(icon, color: Colors.white, size: 30),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}

class DetectionResult {
  final String label;
  final double confidence;
  final Rect boundingBox;

  DetectionResult({
    required this.label,
    required this.confidence,
    required this.boundingBox,
  });
}

class BoundingBoxPainter extends CustomPainter {
  final List<DetectionResult> detections;

  BoundingBoxPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty) return;

    // Use a single detection (the first one)
    var detection = detections[0];
    var color = Colors.green; // Use green for detection indicator

    // Create a more prominent paint for the detection indicator
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0;

    // Convert normalized coordinates to actual pixel coordinates
    final rect = Rect.fromLTWH(
      detection.boundingBox.left * size.width,
      detection.boundingBox.top * size.height,
      detection.boundingBox.width * size.width,
      detection.boundingBox.height * size.height,
    );

    // Draw the detection indicator box
    canvas.drawRect(rect, paint);

    // Draw corner indicators for better visibility
    final cornerPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill
      ..strokeWidth = 2.0;

    double cornerSize = 20.0;

    // Top-left corner
    canvas.drawRect(
      Rect.fromLTWH(rect.left, rect.top, cornerSize, 4),
      cornerPaint,
    );
    canvas.drawRect(
      Rect.fromLTWH(rect.left, rect.top, 4, cornerSize),
      cornerPaint,
    );

    // Top-right corner
    canvas.drawRect(
      Rect.fromLTWH(rect.right - cornerSize, rect.top, cornerSize, 4),
      cornerPaint,
    );
    canvas.drawRect(
      Rect.fromLTWH(rect.right - 4, rect.top, 4, cornerSize),
      cornerPaint,
    );

    // Bottom-left corner
    canvas.drawRect(
      Rect.fromLTWH(rect.left, rect.bottom - 4, cornerSize, 4),
      cornerPaint,
    );
    canvas.drawRect(
      Rect.fromLTWH(rect.left, rect.bottom - cornerSize, 4, cornerSize),
      cornerPaint,
    );

    // Bottom-right corner
    canvas.drawRect(
      Rect.fromLTWH(rect.right - cornerSize, rect.bottom - 4, cornerSize, 4),
      cornerPaint,
    );
    canvas.drawRect(
      Rect.fromLTWH(rect.right - 4, rect.bottom - cornerSize, 4, cornerSize),
      cornerPaint,
    );

    // Draw label background
    final textSpan = TextSpan(
      text:
          '${detection.label} ${(detection.confidence * 100).toStringAsFixed(1)}%',
      style: const TextStyle(
        color: Colors.white,
        fontSize: 16,
        fontWeight: FontWeight.bold,
      ),
    );
    final textPainter = TextPainter(
      text: textSpan,
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();

    final textRect = Rect.fromLTWH(
      rect.left,
      rect.top - textPainter.height - 10,
      textPainter.width + 20,
      textPainter.height + 10,
    );

    // Draw label background with rounded corners
    final labelPaint = Paint()
      ..color = color.withOpacity(0.9)
      ..style = PaintingStyle.fill;

    final labelRect = RRect.fromRectAndRadius(
      textRect,
      const Radius.circular(8),
    );
    canvas.drawRRect(labelRect, labelPaint);

    // Draw text
    textPainter.paint(
      canvas,
      Offset(rect.left + 10, rect.top - textPainter.height - 5),
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
