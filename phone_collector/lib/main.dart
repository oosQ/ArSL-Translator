import 'dart:convert';
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
  static const int _predictionHistorySize = 6;
  static const Map<String, String> _arabicLabelNames = {
    'alif': 'حرف الألف',
    'ba': 'حرف الباء',
    'ta': 'حرف التاء',
    'tha': 'حرف الثاء',
    'jeem': 'حرف الجيم',
    'ha': 'حرف الحاء',
    'kha': 'حرف الخاء',
    'dal': 'حرف الدال',
    'thal': 'حرف الذال',
    'raa': 'حرف الراء',
    'unknown': 'إشارة غير معروفة',
    'Unknown sign': 'إشارة غير معروفة',
  };

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
  String _collectorStatus = 'Starting camera...';
  String _detectorStatus = 'Loading trained model...';
  GestureModel? _gestureModel;
  GesturePrediction? _prediction;
  DatasetTableSource? _tableSource;
  final List<GesturePrediction> _predictionHistory = [];

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

    await _loadBundledModel();
    await _setupCamera();
  }

  String _displayLabel(String label) {
    return _arabicLabelNames[label] ?? label;
  }

  Future<void> _loadBundledModel() async {
    try {
      final jsonString = await rootBundle.loadString(
        'assets/models/gesture_model.json',
      );
      final model = GestureModel.fromJson(
        jsonDecode(jsonString) as Map<String, dynamic>,
      );

      if (!mounted) {
        return;
      }

      setState(() {
        _gestureModel = model;
        final rawPrediction = _hands.isEmpty
            ? null
            : model.predict(_featureVectorFromLandmarks(_hands.first.landmarks));
        _prediction = _smoothedPrediction(rawPrediction);
        _detectorStatus =
            'Loaded trained model with ${(100 * model.testAccuracy).toStringAsFixed(1)}% test accuracy.';
      });
    } catch (e) {
      if (!mounted) {
        return;
      }

      setState(() {
        _gestureModel = null;
        _prediction = null;
        _detectorStatus = 'Could not load trained model: $e';
      });
    }
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
      final smoothedPrediction = _smoothedPrediction(prediction);

      if (mounted) {
        setState(() {
          _hands = hands;
          _prediction = smoothedPrediction;

          if (hands.isEmpty) {
            _collectorStatus = 'No hand detected.';
            _detectorStatus = _gestureModel == null
                ? 'No trained model loaded.'
                : 'No hand detected. Show one of the trained signs.';
          } else {
            _collectorStatus = 'Hand detected. Tap Save Sample.';
            _detectorStatus = smoothedPrediction == null
                ? 'Trained model loaded. Show one of the trained signs.'
                : 'Detected ${_displayLabel(smoothedPrediction.label)} (${(smoothedPrediction.confidence * 100).toStringAsFixed(1)}%).';
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
          _collectorStatus =
              'Saved sample #$_savedCount for "$label" to ${file.path}.';
          _detectorStatus =
              'Bundled model still loaded. Retrain offline if you want it to learn new samples.';
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

  void _showRowDetails(
    BuildContext context,
    List<String> headers,
    List<String> row,
    int rowIndex,
  ) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (context) => SafeArea(
        child: DraggableScrollableSheet(
          expand: false,
          initialChildSize: 0.8,
          minChildSize: 0.5,
          maxChildSize: 0.95,
          builder: (context, scrollController) => Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'Row ${rowIndex + 1}',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 8),
                Expanded(
                  child: ListView.separated(
                    controller: scrollController,
                    itemCount: headers.length,
                    separatorBuilder: (_, __) => const Divider(height: 1),
                    itemBuilder: (context, index) => ListTile(
                      dense: true,
                      title: Text(headers[index]),
                      subtitle: Text(index < row.length ? row[index] : ''),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
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
          _predictionHistory.clear();
          _prediction = null;
          _collectorStatus = 'Deleted dataset file at ${file.path}.';
          _detectorStatus =
              'Dataset deleted. The bundled trained model is still available.';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _collectorStatus = 'Delete failed: $e');
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
      _predictionHistory.clear();
      _prediction = null;
      _cameraIndex = (_cameraIndex + 1) % widget.cameras.length;
      _collectorStatus = 'Switching camera...';
      _detectorStatus = 'Switching camera...';
    });

    await _setupCamera();
  }

  List<double> _featureVectorFromLandmarks(List<dynamic> landmarks) {
    final rawValues = <double>[];
    for (final landmark in landmarks) {
      rawValues.add((landmark.x as num).toDouble());
      rawValues.add((landmark.y as num).toDouble());
      rawValues.add((landmark.z as num).toDouble());
    }
    final points = _normalizePoints(rawValues);
    final features = <double>[
      for (final point in points) ...point,
      for (final (start, end) in GestureModel.distancePairs)
        _pointDistance(points[start], points[end]),
      for (final (a, b, c) in GestureModel.angleTriplets)
        _jointAngle(points[a], points[b], points[c]),
    ];
    return features;
  }

  List<List<double>> _normalizePoints(List<double> values) {
    if (values.length < 63) {
      return const [];
    }

    final wristX = values[0];
    final wristY = values[1];
    final wristZ = values[2];
    final normalized = <List<double>>[];
    var maxDistance = 0.0;

    for (var i = 0; i < 21; i++) {
      final dx = values[(i * 3)] - wristX;
      final dy = values[(i * 3) + 1] - wristY;
      final dz = values[(i * 3) + 2] - wristZ;
      normalized.add([dx, dy, dz]);
      final distance = _pointDistance([0, 0, 0], [dx, dy, dz]);
      if (distance > maxDistance) {
        maxDistance = distance;
      }
    }

    final scale = maxDistance == 0 ? 1.0 : maxDistance;
    return normalized
        .map((point) => point.map((value) => value / scale).toList())
        .toList();
  }

  double _pointDistance(List<double> a, List<double> b) {
    final dx = a[0] - b[0];
    final dy = a[1] - b[1];
    final dz = a[2] - b[2];
    return math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  double _jointAngle(List<double> a, List<double> b, List<double> c) {
    final ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    final cb = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    final abNorm = math.sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);
    final cbNorm = math.sqrt(cb[0] * cb[0] + cb[1] * cb[1] + cb[2] * cb[2]);
    if (abNorm == 0 || cbNorm == 0) {
      return 0;
    }

    final dot = (ab[0] * cb[0]) + (ab[1] * cb[1]) + (ab[2] * cb[2]);
    final cosine = (dot / (abNorm * cbNorm)).clamp(-1.0, 1.0);
    return math.acos(cosine) / math.pi;
  }

  GesturePrediction? _smoothedPrediction(GesturePrediction? prediction) {
    if (prediction == null) {
      _predictionHistory.clear();
      return null;
    }

    _predictionHistory.add(prediction);
    if (_predictionHistory.length > _predictionHistorySize) {
      _predictionHistory.removeAt(0);
    }

    final grouped = <String, List<GesturePrediction>>{};
    for (final item in _predictionHistory) {
      grouped.putIfAbsent(item.label, () => <GesturePrediction>[]).add(item);
    }

    var bestLabel = prediction.label;
    var bestGroup = grouped[prediction.label] ?? <GesturePrediction>[prediction];

    for (final entry in grouped.entries) {
      if (entry.value.length > bestGroup.length) {
        bestLabel = entry.key;
        bestGroup = entry.value;
      }
    }

    final averageConfidence =
        bestGroup.fold<double>(0, (sum, item) => sum + item.confidence) /
        bestGroup.length;
    final averageScore =
        bestGroup.fold<double>(0, (sum, item) => sum + item.score) /
        bestGroup.length;
    final unknownVotes = bestGroup.where((item) => item.isUnknown).length;
    final isUnknown = unknownVotes >= ((bestGroup.length + 1) ~/ 2);

    return GesturePrediction(
      label: isUnknown ? 'Unknown sign' : bestLabel,
      score: averageScore,
      confidence: averageConfidence,
      isUnknown: isUnknown,
    );
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
        Flexible(
          child: SafeArea(
            top: false,
            minimum: const EdgeInsets.fromLTRB(16, 16, 16, 16),
            child: SingleChildScrollView(
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
          ),
        ),
      ],
    );
  }

  Widget _buildTableHeader(BuildContext context, DatasetSnapshot dataset) {
    final dataRows = dataset.rows.skip(1).toList();
    return Padding(
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
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 8),
          Text(
            'Tap any row to see all CSV values.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ),
    );
  }

  Widget _buildTableWidget(BuildContext context, DatasetSnapshot dataset) {
    final headers = dataset.rows.first;
    final dataRows = dataset.rows.skip(1).toList();

    _tableSource = DatasetTableSource(
      rows: dataRows,
      onRowTap: (rowIndex) {
        _showRowDetails(context, headers, dataRows[rowIndex], rowIndex);
      },
    );

    return Expanded(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(12, 0, 12, 16),
        children: [
          PaginatedDataTable(
            header: const Text('Samples'),
            rowsPerPage: 12,
            showCheckboxColumn: false,
            columnSpacing: 16,
            horizontalMargin: 12,
            columns: const [
              DataColumn(label: Text('#')),
              DataColumn(label: Text('Label')),
              DataColumn(label: Text('x0')),
              DataColumn(label: Text('y0')),
              DataColumn(label: Text('z0')),
              DataColumn(label: Text('x8')),
              DataColumn(label: Text('y8')),
              DataColumn(label: Text('z8')),
            ],
            source: _tableSource!,
          ),
        ],
      ),
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

        final dataRows = dataset.rows.skip(1).toList();

        return SafeArea(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildTableHeader(context, dataset),
              const Divider(height: 1),
              if (dataRows.isEmpty)
                const Expanded(
                  child: Center(
                    child: Text('The file only contains the header row.'),
                  ),
                )
              else
                _buildTableWidget(context, dataset),
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
        Flexible(
          child: SafeArea(
            top: false,
            minimum: const EdgeInsets.fromLTRB(16, 16, 16, 16),
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  FilledButton.icon(
                    onPressed: _loadBundledModel,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Reload Trained Model'),
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
                                : _displayLabel(_prediction!.label),
                            style: Theme.of(context).textTheme.headlineSmall,
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _prediction == null
                                ? 'Show one of the trained signs to run inference.'
                                : _prediction!.isUnknown
                                    ? 'Confidence ${(100 * _prediction!.confidence).toStringAsFixed(1)}% | unknown threshold triggered'
                                    : 'Confidence ${(100 * _prediction!.confidence).toStringAsFixed(1)}% | score ${_prediction!.score.toStringAsFixed(3)}',
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
                            'Model Metrics',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          const SizedBox(height: 8),
                          if (_gestureModel == null)
                            const Text('No trained model loaded.')
                          else
                            ...[
                              Text(
                                'Test accuracy: ${(100 * _gestureModel!.testAccuracy).toStringAsFixed(1)}%',
                              ),
                              Text(
                                'Train accuracy: ${(100 * _gestureModel!.trainAccuracy).toStringAsFixed(1)}%',
                              ),
                              Text('Samples: ${_gestureModel!.sampleCount}'),
                              Text(
                                'Unknown threshold: ${(100 * _gestureModel!.unknownConfidenceThreshold).toStringAsFixed(0)}% confidence',
                              ),
                              const SizedBox(height: 10),
                              Text(
                                'Labels',
                                style: Theme.of(context).textTheme.titleSmall,
                              ),
                              const SizedBox(height: 8),
                              ..._gestureModel!.labelCounts.entries.map(
                                (entry) => Padding(
                                  padding: const EdgeInsets.only(bottom: 6),
                                  child: Text(
                                    '${_displayLabel(entry.key)}: ${entry.value} samples',
                                  ),
                                ),
                              ),
                            ],
                        ],
                      ),
                    ),
                  ),
                ],
              ),
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

class DatasetTableSource extends DataTableSource {
  DatasetTableSource({
    required this.rows,
    required this.onRowTap,
  });

  final List<List<String>> rows;
  final ValueChanged<int> onRowTap;

  String _valueAt(List<String> row, int index) {
    if (index >= row.length) {
      return '';
    }
    final value = row[index];
    if (index == 63) {
      return value;
    }
    final parsed = double.tryParse(value);
    if (parsed == null) {
      return value;
    }
    return parsed.toStringAsFixed(3);
  }

  @override
  DataRow? getRow(int index) {
    if (index >= rows.length) {
      return null;
    }

    final row = rows[index];
    return DataRow.byIndex(
      index: index,
      onSelectChanged: (_) => onRowTap(index),
      cells: [
        DataCell(Text('${index + 1}')),
        DataCell(Text(_valueAt(row, 63))),
        DataCell(Text(_valueAt(row, 0))),
        DataCell(Text(_valueAt(row, 1))),
        DataCell(Text(_valueAt(row, 2))),
        DataCell(Text(_valueAt(row, 24))),
        DataCell(Text(_valueAt(row, 25))),
        DataCell(Text(_valueAt(row, 26))),
      ],
    );
  }

  @override
  bool get isRowCountApproximate => false;

  @override
  int get rowCount => rows.length;

  @override
  int get selectedRowCount => 0;
}

class GestureModel {
  const GestureModel({
    required this.labels,
    required this.layers,
    required this.featureMean,
    required this.featureStd,
    required this.trainAccuracy,
    required this.testAccuracy,
    required this.sampleCount,
    required this.labelCounts,
    required this.unknownConfidenceThreshold,
    required this.unknownMarginThreshold,
  });

  static const List<(int, int)> distancePairs = [
    (0, 4),
    (0, 8),
    (0, 12),
    (0, 16),
    (0, 20),
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (5, 9),
    (9, 13),
    (13, 17),
    (5, 17),
  ];

  static const List<(int, int, int)> angleTriplets = [
    (1, 2, 3),
    (2, 3, 4),
    (5, 6, 7),
    (6, 7, 8),
    (9, 10, 11),
    (10, 11, 12),
    (13, 14, 15),
    (14, 15, 16),
    (17, 18, 19),
    (18, 19, 20),
  ];

  final List<String> labels;
  final List<ModelLayer> layers;
  final List<double> featureMean;
  final List<double> featureStd;
  final double trainAccuracy;
  final double testAccuracy;
  final int sampleCount;
  final Map<String, int> labelCounts;
  final double unknownConfidenceThreshold;
  final double unknownMarginThreshold;

  factory GestureModel.fromJson(Map<String, dynamic> json) {
    final metrics = json['metrics'] as Map<String, dynamic>? ?? {};
    final rawLabelCounts = metrics['label_counts'] as Map<String, dynamic>? ?? {};
    final thresholds = json['thresholds'] as Map<String, dynamic>? ?? {};

    return GestureModel(
      labels: (json['labels'] as List<dynamic>).cast<String>(),
      layers: (json['layers'] as List<dynamic>)
          .map((layer) => ModelLayer.fromJson(layer as Map<String, dynamic>))
          .toList(),
      featureMean: (json['feature_mean'] as List<dynamic>)
          .map((v) => (v as num).toDouble())
          .toList(),
      featureStd: (json['feature_std'] as List<dynamic>)
          .map((v) => (v as num).toDouble())
          .toList(),
      trainAccuracy: (metrics['train_accuracy'] as num?)?.toDouble() ?? 0,
      testAccuracy: (metrics['test_accuracy'] as num?)?.toDouble() ?? 0,
      sampleCount: (metrics['sample_count'] as num?)?.toInt() ?? 0,
      labelCounts: rawLabelCounts.map(
        (key, value) => MapEntry(key, (value as num).toInt()),
      ),
      unknownConfidenceThreshold:
          (thresholds['unknown_confidence'] as num?)?.toDouble() ?? 0.70,
      unknownMarginThreshold:
          (thresholds['unknown_margin'] as num?)?.toDouble() ?? 0.18,
    );
  }

  GesturePrediction? predict(List<double> features) {
    if (features.isEmpty ||
        labels.isEmpty ||
        layers.isEmpty ||
        featureMean.length != features.length ||
        featureStd.length != features.length) {
      return null;
    }

    final standardized = List<double>.generate(features.length, (index) {
      final std = featureStd[index].abs() < 1e-8 ? 1.0 : featureStd[index];
      return (features[index] - featureMean[index]) / std;
    });

    var activations = standardized;
    for (final layer in layers) {
      activations = layer.forward(activations);
    }

    if (activations.length != labels.length) {
      return null;
    }

    final maxLogit = activations.reduce(math.max);
    final expScores =
        activations.map((logit) => math.exp(logit - maxLogit)).toList();
    final totalScore = expScores.reduce((sum, value) => sum + value);
    final probabilities = expScores.map((value) => value / totalScore).toList();

    var bestIndex = 0;
    for (var i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[bestIndex]) {
        bestIndex = i;
      }
    }

    final sorted = [...probabilities]..sort();
    final bestProbability = probabilities[bestIndex];
    final secondBestProbability =
        sorted.length > 1 ? sorted[sorted.length - 2] : 0.0;
    final isUnknown =
        bestProbability < unknownConfidenceThreshold ||
        (bestProbability - secondBestProbability) < unknownMarginThreshold;

    return GesturePrediction(
      label: isUnknown ? 'Unknown sign' : labels[bestIndex],
      score: activations[bestIndex],
      confidence: bestProbability,
      isUnknown: isUnknown,
    );
  }
}

class ModelLayer {
  const ModelLayer({
    required this.activation,
    required this.weights,
    required this.bias,
  });

  final String activation;
  final List<List<double>> weights;
  final List<double> bias;

  factory ModelLayer.fromJson(Map<String, dynamic> json) {
    return ModelLayer(
      activation: json['activation'] as String? ?? 'linear',
      weights: (json['weights'] as List<dynamic>)
          .map(
            (row) => (row as List<dynamic>)
                .map((value) => (value as num).toDouble())
                .toList(),
          )
          .toList(),
      bias: (json['bias'] as List<dynamic>)
          .map((value) => (value as num).toDouble())
          .toList(),
    );
  }

  List<double> forward(List<double> input) {
    final output = List<double>.filled(bias.length, 0);
    for (var outputIndex = 0; outputIndex < bias.length; outputIndex++) {
      var total = bias[outputIndex];
      for (var inputIndex = 0; inputIndex < input.length; inputIndex++) {
        total += input[inputIndex] * weights[inputIndex][outputIndex];
      }
      output[outputIndex] = activation == 'relu' ? math.max(0, total) : total;
    }
    return output;
  }
}

class GesturePrediction {
  const GesturePrediction({
    required this.label,
    required this.score,
    required this.confidence,
    required this.isUnknown,
  });

  final String label;
  final double score;
  final double confidence;
  final bool isUnknown;
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
