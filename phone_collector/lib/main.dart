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
  int _selectedIndex = 0;
  int _datasetVersion = 0;
  bool _isInitialized = false;
  bool _isDetecting = false;
  bool _isSaving = false;
  bool _isBuildingModel = false;
  String _collectorStatus = 'Starting camera...';
  String _detectorStatus = 'Build a model from your collected samples.';
  GestureModel? _gestureModel;
  GesturePrediction? _prediction;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    if (widget.cameras.isEmpty) {
      setState(() => _collectorStatus = 'No camera found on this phone.');
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
          _collectorStatus = 'Show your hand, enter a label, then tap Save Sample.';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _collectorStatus = 'Could not open camera: $e');
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

      final prediction = hands.isEmpty || _gestureModel == null
          ? null
          : _gestureModel!.predict(_featureVectorFromLandmarks(hands.first.landmarks));

      if (mounted) {
        setState(() {
          _hands = hands;
          _prediction = prediction;

          if (hands.isEmpty) {
            _collectorStatus = 'No hand detected.';
            _detectorStatus = _gestureModel == null
                ? 'Build a model from your collected samples.'
                : 'No hand detected. Show one of the trained signs.';
          } else {
            _collectorStatus = 'Hand detected. Tap Save Sample.';
            _detectorStatus = prediction == null
                ? 'Model ready. Show one of the trained signs.'
                : 'Detected ${prediction.label} (${(prediction.confidence * 100).toStringAsFixed(1)}%).';
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _collectorStatus = 'Detection error: $e';
          _detectorStatus = 'Detection error: $e';
        });
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
      setState(() => _collectorStatus = 'Label cannot be empty.');
      return;
    }

    if (_hands.isEmpty || _hands.first.landmarks.length != 21) {
      setState(
        () => _collectorStatus = 'No hand landmarks ready. Show your hand first.',
      );
      return;
    }

    setState(() {
      _isSaving = true;
      _collectorStatus = 'Saving sample...';
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
          _datasetVersion += 1;
          _gestureModel = null;
          _prediction = null;
          _collectorStatus =
              'Saved sample #$_savedCount for "$label" to ${file.path}.';
          _detectorStatus = 'Dataset changed. Build the model again.';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _collectorStatus = 'Save failed: $e');
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
      setState(() => _collectorStatus = 'No dataset file yet at ${file.path}.');
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

  Future<DatasetSnapshot> _loadDatasetSnapshot() async {
    final file = await _datasetFile();
    final rows = await _readDatasetRows();
    return DatasetSnapshot(filePath: file.path, rows: rows);
  }

  Future<void> _deleteDataset() async {
    final file = await _datasetFile();
    if (!await file.exists()) {
      if (mounted) {
        setState(() {
          _collectorStatus = 'No dataset file to delete at ${file.path}.';
        });
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
          _datasetVersion += 1;
          _gestureModel = null;
          _prediction = null;
          _collectorStatus = 'Deleted dataset file at ${file.path}.';
          _detectorStatus =
              'Dataset deleted. Collect samples, then build the model again.';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _collectorStatus = 'Delete failed: $e');
      }
    }
  }

  Future<void> _buildGestureModel() async {
    setState(() {
      _isBuildingModel = true;
      _detectorStatus = 'Building model from dataset.csv...';
    });

    try {
      final rows = await _readDatasetRows();
      if (rows.length <= 1) {
        setState(() {
          _gestureModel = null;
          _prediction = null;
          _detectorStatus = 'No training samples yet. Collect data first.';
        });
        return;
      }

      final groupedFeatures = <String, List<List<double>>>{};
      for (final row in rows.skip(1)) {
        if (row.length < 64) {
          continue;
        }

        final values = <double>[];
        var valid = true;
        for (var i = 0; i < 63; i++) {
          final parsed = double.tryParse(row[i]);
          if (parsed == null) {
            valid = false;
            break;
          }
          values.add(parsed);
        }
        if (!valid) {
          continue;
        }

        final label = row[63].trim();
        if (label.isEmpty) {
          continue;
        }

        groupedFeatures
            .putIfAbsent(label, () => <List<double>>[])
            .add(_normalizeFeatureVector(values));
      }

      if (groupedFeatures.isEmpty) {
        setState(() {
          _gestureModel = null;
          _prediction = null;
          _detectorStatus = 'Dataset rows are invalid. Recollect the samples.';
        });
        return;
      }

      final centroids = <String, List<double>>{};
      final sampleCounts = <String, int>{};

      for (final entry in groupedFeatures.entries) {
        final samples = entry.value;
        final centroid = List<double>.filled(samples.first.length, 0);
        for (final sample in samples) {
          for (var i = 0; i < sample.length; i++) {
            centroid[i] += sample[i];
          }
        }
        for (var i = 0; i < centroid.length; i++) {
          centroid[i] /= samples.length;
        }
        centroids[entry.key] = centroid;
        sampleCounts[entry.key] = samples.length;
      }

      setState(() {
        _gestureModel = GestureModel(
          centroids: centroids,
          sampleCounts: sampleCounts,
        );
        _prediction = _hands.isEmpty
            ? null
            : _gestureModel!.predict(_featureVectorFromLandmarks(_hands.first.landmarks));
        _detectorStatus =
            'Model built for ${centroids.length} signs from ${rows.length - 1} samples.';
      });
    } catch (e) {
      setState(() {
        _gestureModel = null;
        _prediction = null;
        _detectorStatus = 'Model build failed: $e';
      });
    } finally {
      if (mounted) {
        setState(() => _isBuildingModel = false);
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
      _prediction = null;
      _cameraIndex = (_cameraIndex + 1) % widget.cameras.length;
      _collectorStatus = 'Switching camera...';
      _detectorStatus = 'Switching camera...';
    });

    await _setupCamera();
  }

  List<double> _featureVectorFromLandmarks(List<dynamic> landmarks) {
    final values = <double>[];
    for (final landmark in landmarks) {
      values.add((landmark.x as num).toDouble());
      values.add((landmark.y as num).toDouble());
      values.add((landmark.z as num).toDouble());
    }
    return _normalizeFeatureVector(values);
  }

  List<double> _normalizeFeatureVector(List<double> values) {
    if (values.length < 63) {
      return values;
    }

    final wristX = values[0];
    final wristY = values[1];
    final wristZ = values[2];
    final normalized = <double>[];
    var maxDistance = 0.0;

    for (var i = 0; i < 21; i++) {
      final dx = values[(i * 3)] - wristX;
      final dy = values[(i * 3) + 1] - wristY;
      final dz = values[(i * 3) + 2] - wristZ;
      normalized.addAll([dx, dy, dz]);
      final distance = math.sqrt(dx * dx + dy * dy + dz * dz);
      if (distance > maxDistance) {
        maxDistance = distance;
      }
    }

    final scale = maxDistance == 0 ? 1.0 : maxDistance;
    return normalized.map((value) => value / scale).toList();
  }

  String get _pageTitle {
    switch (_selectedIndex) {
      case 1:
        return 'Table View';
      case 2:
        return 'Detector Model';
      default:
        return 'Data Collector';
    }
  }

  List<Widget> _buildActions() {
    switch (_selectedIndex) {
      case 1:
        return [
          IconButton(
            onPressed: () => setState(() => _datasetVersion += 1),
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh dataset',
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
        ];
      case 2:
        return [
          IconButton(
            onPressed: _switchCamera,
            icon: const Icon(Icons.cameraswitch),
            tooltip: 'Switch camera',
          ),
        ];
      default:
        return [
          IconButton(
            onPressed: _switchCamera,
            icon: const Icon(Icons.cameraswitch),
            tooltip: 'Switch camera',
          ),
        ];
    }
  }

  Widget _buildCameraPreview() {
    final controller = _cameraController;
    final previewSize = controller?.value.previewSize;

    if (!_isInitialized || controller == null || previewSize == null) {
      return const Center(child: CircularProgressIndicator());
    }

    return Center(
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
                  lensDirection: controller.description.lensDirection,
                  sensorOrientation: controller.description.sensorOrientation,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCollectorTab() {
    return Column(
      children: [
        Expanded(child: _buildCameraPreview()),
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
              Text(_collectorStatus),
              const SizedBox(height: 8),
              Text('Saved samples: $_savedCount'),
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
    );
  }

  Widget _buildTableTab() {
    return FutureBuilder<DatasetSnapshot>(
      key: ValueKey(_datasetVersion),
      future: _loadDatasetSnapshot(),
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }

        if (snapshot.hasError) {
          return Center(child: Text('Could not read dataset: ${snapshot.error}'));
        }

        final dataset = snapshot.data;
        if (dataset == null || dataset.rows.isEmpty) {
          return Center(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Text(
                'No dataset file yet. Save some samples in Data Collector first.',
                style: Theme.of(context).textTheme.bodyLarge,
                textAlign: TextAlign.center,
              ),
            ),
          );
        }

        final headers = dataset.rows.first;
        final dataRows = dataset.rows.skip(1).toList();

        return SafeArea(
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
                      '${dataRows.length} data rows',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      dataset.filePath,
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
                                  Theme.of(context)
                                      .colorScheme
                                      .surfaceContainerHighest,
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
                                                index < row.length ? row[index] : '',
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
        );
      },
    );
  }

  Widget _buildDetectorTab() {
    return Column(
      children: [
        Expanded(child: _buildCameraPreview()),
        SafeArea(
          top: false,
          minimum: const EdgeInsets.fromLTRB(16, 16, 16, 16),
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                FilledButton.icon(
                  onPressed: _isBuildingModel ? null : _buildGestureModel,
                  icon: _isBuildingModel
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.model_training),
                  label: const Text('Build Model From CSV'),
                ),
                const SizedBox(height: 12),
                Text(_detectorStatus),
                const SizedBox(height: 12),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Live Prediction',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          _prediction == null
                              ? 'No prediction yet'
                              : _prediction!.label,
                          style: Theme.of(context).textTheme.headlineSmall,
                        ),
                        const SizedBox(height: 4),
                        Text(
                          _prediction == null
                              ? 'Build the model, then show a trained sign.'
                              : 'Confidence ${(100 * _prediction!.confidence).toStringAsFixed(1)}%',
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Trained Labels',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 8),
                        if (_gestureModel == null)
                          const Text('No model built yet.')
                        else
                          ..._gestureModel!.sampleCounts.entries.map(
                            (entry) => Padding(
                              padding: const EdgeInsets.only(bottom: 6),
                              child: Text('${entry.key}: ${entry.value} samples'),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
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
    final pages = <Widget>[
      _buildCollectorTab(),
      _buildTableTab(),
      _buildDetectorTab(),
    ];

    return Scaffold(
      appBar: AppBar(
        title: Text(_pageTitle),
        actions: _buildActions(),
      ),
      body: pages[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) {
          setState(() {
            _selectedIndex = index;
            if (index == 1) {
              _datasetVersion += 1;
            }
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.camera_alt_outlined),
            selectedIcon: Icon(Icons.camera_alt),
            label: 'Collector',
          ),
          NavigationDestination(
            icon: Icon(Icons.table_chart_outlined),
            selectedIcon: Icon(Icons.table_chart),
            label: 'Table View',
          ),
          NavigationDestination(
            icon: Icon(Icons.psychology_outlined),
            selectedIcon: Icon(Icons.psychology),
            label: 'Detector',
          ),
        ],
      ),
    );
  }
}

class DatasetSnapshot {
  const DatasetSnapshot({
    required this.filePath,
    required this.rows,
  });

  final String filePath;
  final List<List<String>> rows;
}

class GestureModel {
  const GestureModel({
    required this.centroids,
    required this.sampleCounts,
  });

  final Map<String, List<double>> centroids;
  final Map<String, int> sampleCounts;

  GesturePrediction? predict(List<double> features) {
    if (features.isEmpty || centroids.isEmpty) {
      return null;
    }

    String? bestLabel;
    double? bestDistance;

    for (final entry in centroids.entries) {
      final centroid = entry.value;
      if (centroid.length != features.length) {
        continue;
      }

      var sum = 0.0;
      for (var i = 0; i < features.length; i++) {
        final delta = features[i] - centroid[i];
        sum += delta * delta;
      }

      final distance = math.sqrt(sum);
      if (bestDistance == null || distance < bestDistance) {
        bestDistance = distance;
        bestLabel = entry.key;
      }
    }

    if (bestLabel == null || bestDistance == null) {
      return null;
    }

    return GesturePrediction(
      label: bestLabel,
      distance: bestDistance,
      confidence: 1 / (1 + bestDistance),
    );
  }
}

class GesturePrediction {
  const GesturePrediction({
    required this.label,
    required this.distance,
    required this.confidence,
  });

  final String label;
  final double distance;
  final double confidence;
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
