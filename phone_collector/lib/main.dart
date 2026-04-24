import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  final cameras = await availableCameras();
  runApp(CollectorApp(cameras: cameras));
}

class CollectorApp extends StatelessWidget {
  const CollectorApp({super.key, required this.cameras});

  final List<CameraDescription> cameras;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Landmark Collector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: CollectorPage(cameras: cameras),
    );
  }
}

class CollectorPage extends StatefulWidget {
  const CollectorPage({super.key, required this.cameras});

  final List<CameraDescription> cameras;

  @override
  State<CollectorPage> createState() => _CollectorPageState();
}

class _CollectorPageState extends State<CollectorPage> {
  final TextEditingController _labelController = TextEditingController();

  CameraController? _cameraController;
  HandLandmarkerPlugin? _handLandmarker;
  List<Hand> _hands = [];

  int _cameraIndex = 0;
  int _savedCount = 0;
  bool _isInitialized = false;
  bool _isDetecting = false;
  bool _isSaving = false;
  String _status = 'Starting camera...';

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    if (widget.cameras.isEmpty) {
      setState(() => _status = 'No camera found on this phone.');
      return;
    }

    _handLandmarker = HandLandmarkerPlugin.create(
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      delegate: HandLandmarkerDelegate.gpu,
    );

    await _setupCamera();
  }

  Future<void> _setupCamera() async {
    final oldController = _cameraController;
    if (oldController != null) {
      if (oldController.value.isStreamingImages) {
        await oldController.stopImageStream();
      }
      await oldController.dispose();
    }

    final frontCameraIndex = widget.cameras.indexWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
    );
    if (_cameraIndex == 0 && frontCameraIndex >= 0) {
      _cameraIndex = frontCameraIndex;
    }

    final controller = CameraController(
      widget.cameras[_cameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    _cameraController = controller;

    try {
      await controller.initialize();
      await controller.startImageStream(_processCameraImage);

      if (mounted) {
        setState(() {
          _isInitialized = true;
          _status = 'Show your hand, enter a label, then tap Save Sample.';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _status = 'Could not open camera: $e');
      }
    }
  }

  void _processCameraImage(CameraImage image) {
    final controller = _cameraController;
    final landmarker = _handLandmarker;
    if (!_isInitialized ||
        _isDetecting ||
        controller == null ||
        landmarker == null) {
      return;
    }

    _isDetecting = true;
    try {
      final hands = landmarker.detect(
        image,
        controller.description.sensorOrientation,
      );

      if (mounted) {
        setState(() {
          _hands = hands;
          if (hands.isEmpty) {
            _status = 'No hand detected.';
          } else {
            _status = 'Hand detected. Tap Save Sample.';
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _status = 'Detection error: $e');
      }
    } finally {
      _isDetecting = false;
    }
  }

  Future<File> _datasetFile() async {
    final directory = await getApplicationDocumentsDirectory();
    await directory.create(recursive: true);
    return File('${directory.path}/dataset.csv');
  }

  Future<void> _ensureHeader(File file) async {
    if (await file.exists()) {
      return;
    }

    final columns = <String>[];
    for (var i = 0; i < 21; i++) {
      columns.addAll(['x$i', 'y$i', 'z$i']);
    }
    columns.add('label');
    await file.writeAsString('${columns.join(',')}\n', flush: true);
  }

  Future<void> _saveSample() async {
    final label = _labelController.text.trim();
    if (label.isEmpty) {
      setState(() => _status = 'Label cannot be empty.');
      return;
    }

    if (_hands.isEmpty || _hands.first.landmarks.length != 21) {
      setState(
          () => _status = 'No hand landmarks ready. Show your hand first.');
      return;
    }

    setState(() {
      _isSaving = true;
      _status = 'Saving sample...';
    });

    try {
      final file = await _datasetFile();
      await _ensureHeader(file);

      final values = <String>[];
      for (final landmark in _hands.first.landmarks) {
        values.addAll([
          landmark.x.toString(),
          landmark.y.toString(),
          landmark.z.toString(),
        ]);
      }
      values.add(label);

      await file.writeAsString(
        '${values.join(',')}\n',
        mode: FileMode.append,
        flush: true,
      );

      if (mounted) {
        setState(() {
          _savedCount += 1;
          _status =
              'Saved sample #$_savedCount for "$label" to ${file.path}.';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _status = 'Save failed: $e');
      }
    } finally {
      if (mounted) {
        setState(() => _isSaving = false);
      }
    }
  }

  Future<void> _shareDataset() async {
    final file = await _datasetFile();
    if (!await file.exists()) {
      setState(() => _status = 'No dataset file yet at ${file.path}.');
      return;
    }

    await Share.shareXFiles(
      [XFile(file.path)],
      text: 'Collected hand landmark dataset',
    );
  }

  Future<List<List<String>>> _readDatasetRows() async {
    final file = await _datasetFile();
    if (!await file.exists()) {
      return [];
    }

    final lines = await file.readAsLines();
    return lines
        .where((line) => line.trim().isNotEmpty)
        .map((line) => line.split(','))
        .toList();
  }

  Future<void> _openDatasetPage() async {
    final rows = await _readDatasetRows();
    if (!mounted) {
      return;
    }

    if (rows.isEmpty) {
      setState(() => _status = 'No dataset file to view yet.');
      return;
    }

    final file = await _datasetFile();
    if (!mounted) {
      return;
    }

    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (context) => DatasetPage(
          rows: rows,
          filePath: file.path,
        ),
      ),
    );
  }

  Future<void> _deleteDataset() async {
    final file = await _datasetFile();
    if (!await file.exists()) {
      if (mounted) {
        setState(() => _status = 'No dataset file to delete at ${file.path}.');
      }
      return;
    }

    final shouldDelete = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete dataset'),
        content: Text('Remove ${file.path}?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (shouldDelete != true) {
      return;
    }

    try {
      await file.delete();
      if (mounted) {
        setState(() {
          _savedCount = 0;
          _status = 'Deleted dataset file at ${file.path}.';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _status = 'Delete failed: $e');
      }
    }
  }

  Future<void> _switchCamera() async {
    if (widget.cameras.length < 2 || _isSaving) {
      return;
    }

    setState(() {
      _isInitialized = false;
      _hands = [];
      _cameraIndex = (_cameraIndex + 1) % widget.cameras.length;
      _status = 'Switching camera...';
    });

    await _setupCamera();
  }

  @override
  void dispose() {
    _labelController.dispose();
    _cameraController?.stopImageStream();
    _cameraController?.dispose();
    _handLandmarker?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _cameraController;
    final previewSize = controller?.value.previewSize;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Landmark Collector'),
        actions: [
          IconButton(
            onPressed: _switchCamera,
            icon: const Icon(Icons.cameraswitch),
            tooltip: 'Switch camera',
          ),
          IconButton(
            onPressed: _openDatasetPage,
            icon: const Icon(Icons.table_view),
            tooltip: 'View dataset',
          ),
          IconButton(
            onPressed: _shareDataset,
            icon: const Icon(Icons.ios_share),
            tooltip: 'Share dataset',
          ),
          IconButton(
            onPressed: _deleteDataset,
            icon: const Icon(Icons.delete_outline),
            tooltip: 'Delete dataset',
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: !_isInitialized || controller == null || previewSize == null
                ? const Center(child: CircularProgressIndicator())
                : Center(
                    child: AspectRatio(
                      aspectRatio: previewSize.height / previewSize.width,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          CameraPreview(controller),
                          IgnorePointer(
                            child: CustomPaint(
                              painter: LandmarkPainter(
                                hands: _hands,
                                previewSize: previewSize,
                                lensDirection:
                                    controller.description.lensDirection,
                                sensorOrientation:
                                    controller.description.sensorOrientation,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
          ),
          SafeArea(
            top: false,
            minimum: const EdgeInsets.fromLTRB(16, 16, 16, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                TextField(
                  controller: _labelController,
                  textInputAction: TextInputAction.done,
                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    labelText: 'Label',
                    hintText: 'alif, ba, ta...',
                  ),
                ),
                const SizedBox(height: 10),
                Text(_status),
                const SizedBox(height: 12),
                FilledButton.icon(
                  onPressed: _isSaving ? null : _saveSample,
                  icon: _isSaving
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.save),
                  label: const Text('Save Sample'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class DatasetPage extends StatelessWidget {
  const DatasetPage({
    super.key,
    required this.rows,
    required this.filePath,
  });

  final List<List<String>> rows;
  final String filePath;

  @override
  Widget build(BuildContext context) {
    final headers = rows.first;
    final dataRows = rows.skip(1).toList();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Dataset View'),
      ),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'dataset.csv',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${dataRows.length} rows',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    filePath,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ),
            ),
            const Divider(height: 1),
            Expanded(
              child: dataRows.isEmpty
                  ? const Center(
                      child: Text('The file only contains the header row.'),
                    )
                  : Scrollbar(
                      thumbVisibility: true,
                      child: SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        padding: const EdgeInsets.all(12),
                        child: ConstrainedBox(
                          constraints: BoxConstraints(
                            minWidth: MediaQuery.of(context).size.width - 24,
                          ),
                          child: SingleChildScrollView(
                            child: DataTable(
                              headingRowColor: WidgetStatePropertyAll(
                                Theme.of(
                                  context,
                                ).colorScheme.surfaceContainerHighest,
                              ),
                              headingTextStyle: Theme.of(context)
                                  .textTheme
                                  .labelLarge
                                  ?.copyWith(fontWeight: FontWeight.w700),
                              dataRowMinHeight: 44,
                              dataRowMaxHeight: 56,
                              columnSpacing: 20,
                              horizontalMargin: 12,
                              columns: headers
                                  .map(
                                    (header) => DataColumn(
                                      label: SizedBox(
                                        width: 88,
                                        child: Text(
                                          header,
                                          overflow: TextOverflow.ellipsis,
                                        ),
                                      ),
                                    ),
                                  )
                                  .toList(),
                              rows: dataRows
                                  .map(
                                    (row) => DataRow(
                                      cells: List.generate(
                                        headers.length,
                                        (index) => DataCell(
                                          SizedBox(
                                            width: 88,
                                            child: Text(
                                              index < row.length
                                                  ? row[index]
                                                  : '',
                                              maxLines: 2,
                                              overflow: TextOverflow.ellipsis,
                                            ),
                                          ),
                                        ),
                                      ),
                                    ),
                                  )
                                  .toList(),
                            ),
                          ),
                        ),
                      ),
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class LandmarkPainter extends CustomPainter {
  LandmarkPainter({
    required this.hands,
    required this.previewSize,
    required this.lensDirection,
    required this.sensorOrientation,
  });

  final List<Hand> hands;
  final Size previewSize;
  final CameraLensDirection lensDirection;
  final int sensorOrientation;

  static const List<(int, int)> _connections = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final scale = size.width / previewSize.height;

    final connectionPaint = Paint()
      ..color = Colors.white
      ..strokeWidth = 3 / scale
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final pointPaint = Paint()
      ..color = const Color(0xFF00B0FF)
      ..style = PaintingStyle.fill;

    final pointCenterPaint = Paint()
      ..color = const Color(0xFF00FF66)
      ..style = PaintingStyle.fill;

    canvas.save();

    final center = Offset(size.width / 2, size.height / 2);
    canvas.translate(center.dx, center.dy);
    canvas.rotate(sensorOrientation * math.pi / 180);

    if (lensDirection == CameraLensDirection.front) {
      canvas.scale(-1, 1);
      canvas.rotate(math.pi);
    }

    canvas.scale(scale);

    for (final hand in hands) {
      if (hand.landmarks.length < 21) {
        continue;
      }

      final points = hand.landmarks.map((landmark) {
        final x = (landmark.x - 0.5) * previewSize.width;
        final y = (landmark.y - 0.5) * previewSize.height;
        return Offset(x, y);
      }).toList();

      for (final (startIndex, endIndex) in _connections) {
        canvas.drawLine(
          points[startIndex],
          points[endIndex],
          connectionPaint,
        );
      }

      for (final point in points) {
        canvas.drawCircle(point, 6 / scale, pointPaint);
        canvas.drawCircle(point, 2.5 / scale, pointCenterPaint);
      }
    }

    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant LandmarkPainter oldDelegate) {
    return oldDelegate.hands != hands ||
        oldDelegate.previewSize != previewSize ||
        oldDelegate.lensDirection != lensDirection ||
        oldDelegate.sensorOrientation != sensorOrientation;
  }
}
