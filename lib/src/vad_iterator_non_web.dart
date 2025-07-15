// lib/src/vad_iterator_non_web.dart
// ignore_for_file: avoid_print

// Dart imports:
import 'dart:io';
import 'dart:typed_data';

// Flutter imports:
import 'package:flutter/services.dart';

// Package imports:
import 'package:onnxruntime/onnxruntime.dart';

// Project imports:
import 'package:vad/src/vad_iterator.dart';
import 'vad_event.dart';

/// Native platform VAD iterator using ONNX Runtime Flutter for real-time speech detection
class VadIteratorNonWeb implements VadIterator {
  /// Whether to enable debug logging
  final bool _isDebug;

  /// Positive speech threshold
  final double _positiveSpeechThreshold;

  /// Negative speech threshold
  final double _negativeSpeechThreshold;

  /// Number of frames to wait before considering speech as valid
  final int _redemptionFrames;

  /// Frame size in samples - WARNING: Use 512, 1024, or 1536 for best performance
  /// as Silero models were trained on these specific frame sizes
  final int _frameSamples;

  /// Number of frames to pad before speech is considered valid
  final int _preSpeechPadFrames;

  /// Number of frames to pad after speech is considered ended
  final int _endSpeechPadFrames;

  /// Minimum number of speech frames required to consider speech as valid
  final int _minSpeechFrames;

  /// Number of frames to accumulate before emitting chunk
  final int _numFramesToEmit;

  /// Sample rate of the audio stream
  final int _sampleRate;

  /// Silero model version: 'legacy' (v4) or 'v5'
  final String _model;

  /// Whether the user is currently speaking
  bool _speaking = false;

  /// Number of frames since the last speech event
  int _redemptionCounter = 0;

  /// Number of speech frames detected
  int _speechPositiveFrameCount = 0;

  /// Number of samples processed
  int _currentSample = 0;

  /// Buffer for pre-speech frames
  final List<Float32List> _preSpeechBuffer = [];

  /// Buffer for speech frames
  final List<Float32List> _speechBuffer = [];

  /// Index in speech buffer where current speech segment starts
  int _speechStartIndex = 0;

  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;

  static const int _batch = 1;
  var _hide = List.filled(
      2, List.filled(_batch, Float32List.fromList(List.filled(64, 0.0))));
  var _cell = List.filled(
      2, List.filled(_batch, Float32List.fromList(List.filled(64, 0.0))));
  var _state = List.filled(
      2, List.filled(_batch, Float32List.fromList(List.filled(128, 0.0))));

  /// Callback for VAD events
  VadEventCallback? _onVadEvent;

  /// Buffer for audio data
  final List<int> _byteBuffer = [];

  /// Number of bytes per frame
  final int _frameByteCount;

  /// Constructor
  /// [isDebug] - Whether to enable debug logging
  /// [sampleRate] - Sample rate of the audio stream
  /// [frameSamples] - Frame size in samples
  /// [positiveSpeechThreshold] - Positive speech threshold
  /// [negativeSpeechThreshold] - Negative speech threshold
  /// [redemptionFrames] - Number of frames to wait before considering speech as valid
  /// [preSpeechPadFrames] - Number of frames to pad before speech is considered valid
  /// [minSpeechFrames] - Minimum number of speech frames required to consider speech as valid
  /// [model] - Silero model version: 'legacy' (v4) or 'v5'
  /// [baseAssetPath] - Base URL or path for model assets
  /// [endSpeechPadFrames] - Number of frames to pad after speech ends
  /// [numFramesToEmit] - Number of frames to accumulate before emitting chunk
  VadIteratorNonWeb._internal({
    required bool isDebug,
    required int sampleRate,
    required int frameSamples,
    required double positiveSpeechThreshold,
    required double negativeSpeechThreshold,
    required int redemptionFrames,
    required int preSpeechPadFrames,
    required int minSpeechFrames,
    required String model,
    required int endSpeechPadFrames,
    required int numFramesToEmit,
  })  : _isDebug = isDebug,
        _sampleRate = sampleRate,
        _frameSamples = frameSamples,
        _positiveSpeechThreshold = positiveSpeechThreshold,
        _negativeSpeechThreshold = negativeSpeechThreshold,
        _redemptionFrames = redemptionFrames,
        _preSpeechPadFrames = preSpeechPadFrames,
        _minSpeechFrames = minSpeechFrames,
        _model = model,
        _endSpeechPadFrames = endSpeechPadFrames,
        _numFramesToEmit = numFramesToEmit,
        _frameByteCount = frameSamples * 2;

  /// Create and initialize a VadIteratorNonWeb instance
  static Future<VadIteratorNonWeb> create({
    required bool isDebug,
    required int sampleRate,
    required int frameSamples,
    required double positiveSpeechThreshold,
    required double negativeSpeechThreshold,
    required int redemptionFrames,
    required int preSpeechPadFrames,
    required int minSpeechFrames,
    required String model,
    required String baseAssetPath,
    String? onnxWASMBasePath, // Unused on non-web platforms
    int endSpeechPadFrames = 1,
    int numFramesToEmit = 0,
  }) async {
    final instance = VadIteratorNonWeb._internal(
      isDebug: isDebug,
      sampleRate: sampleRate,
      frameSamples: frameSamples,
      positiveSpeechThreshold: positiveSpeechThreshold,
      negativeSpeechThreshold: negativeSpeechThreshold,
      redemptionFrames: redemptionFrames,
      preSpeechPadFrames: preSpeechPadFrames,
      minSpeechFrames: minSpeechFrames,
      model: model,
      endSpeechPadFrames: endSpeechPadFrames,
      numFramesToEmit: numFramesToEmit,
    );

    // Initialize the model
    final modelFile =
        model == 'v5' ? 'silero_vad_v5.onnx' : 'silero_vad_legacy.onnx';
    final modelPath = '$baseAssetPath$modelFile';
    await instance._initModel(modelPath);

    return instance;
  }

  Future<void> _initModel(String modelPath) async {
    try {
      _sessionOptions = OrtSessionOptions()
        ..setInterOpNumThreads(1)
        ..setIntraOpNumThreads(1)
        ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);

      Uint8List bytes;
      if (modelPath.startsWith('http://') || modelPath.startsWith('https://')) {
        final client = HttpClient();
        try {
          final request = await client.getUrl(Uri.parse(modelPath));
          final response = await request.close();
          if (response.statusCode == 200) {
            final completer = BytesBuilder();
            await for (final chunk in response) {
              completer.add(chunk);
            }
            bytes = completer.toBytes();
          } else {
            throw Exception(
                'HTTP ${response.statusCode}: Failed to download model from $modelPath');
          }
        } finally {
          client.close();
        }
      } else {
        // Load from asset bundle (local file)
        final rawAssetFile = await rootBundle.load(modelPath);
        bytes = rawAssetFile.buffer.asUint8List();
      }

      _session = OrtSession.fromBuffer(bytes, _sessionOptions!);
      if (_isDebug) print('VAD model initialized from $modelPath.');
    } catch (e) {
      print('VAD model initialization failed: $e');
      _onVadEvent?.call(VadEvent(
        type: VadEventType.error,
        timestamp: _getCurrentTimestamp(),
        message: 'VAD model initialization failed: $e',
      ));
    }
  }

  @override
  void reset() {
    _speaking = false;
    _redemptionCounter = 0;
    _speechPositiveFrameCount = 0;
    _currentSample = 0;
    _speechStartIndex = 0;
    _preSpeechBuffer.clear();
    _speechBuffer.clear();
    _byteBuffer.clear();
    _hide = List.filled(
        2, List.filled(_batch, Float32List.fromList(List.filled(64, 0.0))));
    _cell = List.filled(
        2, List.filled(_batch, Float32List.fromList(List.filled(64, 0.0))));
    _state = List.filled(
        2, List.filled(_batch, Float32List.fromList(List.filled(128, 0.0))));
  }

  @override
  void release() {
    _sessionOptions?.release();
    _sessionOptions = null;
    _session?.release();
    _session = null;
    OrtEnv.instance.release();
  }

  @override
  void setVadEventCallback(VadEventCallback callback) {
    _onVadEvent = callback;
  }

  @override
  Future<void> processAudioData(Uint8List data) async {
    _byteBuffer.addAll(data);

    while (_byteBuffer.length >= _frameByteCount) {
      final frameBytes = _byteBuffer.sublist(0, _frameByteCount);
      _byteBuffer.removeRange(0, _frameByteCount);
      final frameData = _convertBytesToFloat32(Uint8List.fromList(frameBytes));
      await _processFrame(Float32List.fromList(frameData));
    }
  }

  Future<void> _processFrame(Float32List data) async {
    if (_session == null) {
      print('VAD Iterator: Session not initialized.');
      return;
    }

    final (speechProb, modelOutputs) = await _runModelInference(data);

    // Create a copy of the frame data for the event
    final frameData = data.toList();

    // Emit the frame processed event
    _onVadEvent?.call(VadEvent(
      type: VadEventType.frameProcessed,
      timestamp: _getCurrentTimestamp(),
      message:
          'Frame processed at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
      probabilities: SpeechProbabilities(
          isSpeech: speechProb, notSpeech: 1.0 - speechProb),
      frameData: frameData,
    ));

    for (var element in modelOutputs) {
      element?.release();
    }

    _currentSample += _frameSamples;
    _handleStateTransitions(speechProb, data);
  }

  Future<(double, List<OrtValue?>)> _runModelInference(Float32List data) async {
    if (_model == 'v5') {
      return _runV5ModelInference(data);
    } else {
      return _runLegacyModelInference(data);
    }
  }

  Future<(double, List<OrtValue?>)> _runV5ModelInference(
      Float32List data) async {
    final inputOrt =
        OrtValueTensor.createTensorWithDataList(data, [_batch, _frameSamples]);
    final srOrt = OrtValueTensor.createTensorWithData(_sampleRate);
    final stateOrt = OrtValueTensor.createTensorWithDataList(_state);
    final runOptions = OrtRunOptions();

    final inputs = {'input': inputOrt, 'sr': srOrt, 'state': stateOrt};
    final outputs = _session!.run(runOptions, inputs);

    inputOrt.release();
    srOrt.release();
    stateOrt.release();
    runOptions.release();

    final speechProb = (outputs[0]?.value as List<List<double>>)[0][0];
    _state = (outputs[1]?.value as List<List<List<double>>>)
        .map((e) => e.map((e) => Float32List.fromList(e)).toList())
        .toList();

    return (speechProb, outputs);
  }

  Future<(double, List<OrtValue?>)> _runLegacyModelInference(
      Float32List data) async {
    final inputOrt =
        OrtValueTensor.createTensorWithDataList(data, [_batch, _frameSamples]);
    final srOrt = OrtValueTensor.createTensorWithData(_sampleRate);
    final hOrt = OrtValueTensor.createTensorWithDataList(_hide);
    final cOrt = OrtValueTensor.createTensorWithDataList(_cell);
    final runOptions = OrtRunOptions();

    final inputs = {'input': inputOrt, 'sr': srOrt, 'h': hOrt, 'c': cOrt};
    final outputs = _session!.run(runOptions, inputs);

    inputOrt.release();
    srOrt.release();
    hOrt.release();
    cOrt.release();
    runOptions.release();

    final speechProb = (outputs[0]?.value as List<List<double>>)[0][0];
    _hide = (outputs[1]?.value as List<List<List<double>>>)
        .map((e) => e.map((e) => Float32List.fromList(e)).toList())
        .toList();
    _cell = (outputs[2]?.value as List<List<List<double>>>)
        .map((e) => e.map((e) => Float32List.fromList(e)).toList())
        .toList();

    return (speechProb, outputs);
  }

  void _handleStateTransitions(double speechProb, Float32List data) {
    if (speechProb >= _positiveSpeechThreshold) {
      // Speech-positive frame
      if (!_speaking) {
        _speaking = true;
        _speechStartIndex = 0;
        _onVadEvent?.call(VadEvent(
          type: VadEventType.start,
          timestamp: _getCurrentTimestamp(),
          message:
              'Speech started at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
        ));
        _speechBuffer.addAll(_preSpeechBuffer);
        _preSpeechBuffer.clear();
      }
      _redemptionCounter = 0;
      _speechBuffer.add(data);
      _speechPositiveFrameCount++;

      // Add validation event when speech frames exceed minimum threshold
      if (_speechPositiveFrameCount == _minSpeechFrames) {
        _onVadEvent?.call(VadEvent(
          type: VadEventType.realStart,
          timestamp: _getCurrentTimestamp(),
          message:
              'Speech validated at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
        ));
      }
    } else if (speechProb < _negativeSpeechThreshold) {
      // Handle speech-negative frame
      _handleSpeechNegativeFrame(data);
    } else {
      // Probability between thresholds
      _handleIntermediateFrame(data);
    }

    // Handle chunk emission during speech
    if (_speaking &&
        _numFramesToEmit > 0 &&
        _speechBuffer.length - _speechStartIndex >= _numFramesToEmit &&
        _redemptionCounter <= _endSpeechPadFrames) {
      final framesToSend = _speechBuffer.sublist(
          _speechStartIndex, _speechStartIndex + _numFramesToEmit);
      final audio = _combineSpeechBufferList(framesToSend);
      _speechStartIndex = _speechStartIndex + _numFramesToEmit;
      _onVadEvent?.call(VadEvent(
        type: VadEventType.chunk,
        timestamp: _getCurrentTimestamp(),
        message:
            'Audio chunk emitted at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
        audioData: audio,
      ));
    }
  }

  void _handleSpeechNegativeFrame(Float32List data) {
    if (_speaking) {
      if (++_redemptionCounter >= _redemptionFrames) {
        // End of speech
        _speaking = false;
        _redemptionCounter = 0;

        if (_speechPositiveFrameCount >= _minSpeechFrames) {
          // Valid speech segment
          // Calculate the audio buffer with endSpeechPadFrames
          final frames = _speechBuffer;
          final audioBufferPad = frames.sublist(
              0, frames.length - (_redemptionFrames - _endSpeechPadFrames));
          final audio = _combineSpeechBufferList(audioBufferPad);

          _onVadEvent?.call(VadEvent(
            type: VadEventType.end,
            timestamp: _getCurrentTimestamp(),
            message:
                'Speech ended at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
            audioData: audio,
          ));

          // Handle final chunk emission - emit all accumulated frames since last onEmitChunk
          if (_numFramesToEmit > 0) {
            final speechEndIndex =
                _speechBuffer.length - _redemptionFrames + _endSpeechPadFrames;
            if (_speechStartIndex < speechEndIndex) {
              final framesToSend =
                  _speechBuffer.sublist(_speechStartIndex, speechEndIndex);
              final chunkAudio = _combineSpeechBufferList(framesToSend);
              _onVadEvent?.call(VadEvent(
                type: VadEventType.chunk,
                timestamp: _getCurrentTimestamp(),
                message:
                    'Final audio chunk emitted at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
                audioData: chunkAudio,
              ));
            }
          }
        } else {
          // Misfire
          _onVadEvent?.call(VadEvent(
            type: VadEventType.misfire,
            timestamp: _getCurrentTimestamp(),
            message:
                'Misfire detected at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
          ));
        }
        // Reset counters and buffers
        _speechPositiveFrameCount = 0;
        _speechStartIndex = 0;

        // Preserve frames between endSpeechPadFrames and redemptionFrames for next pre-speech buffer
        if (_endSpeechPadFrames < _redemptionFrames) {
          final framesToKeep = _speechBuffer.sublist(
              _speechBuffer.length - (_redemptionFrames - _endSpeechPadFrames));
          _speechBuffer.clear();
          _preSpeechBuffer.clear();
          _preSpeechBuffer.addAll(framesToKeep);
          // Ensure we don't exceed preSpeechPadFrames limit
          while (_preSpeechBuffer.length > _preSpeechPadFrames) {
            _preSpeechBuffer.removeAt(0);
          }
        } else {
          _speechBuffer.clear();
        }
      } else {
        _speechBuffer.add(data);
      }
    } else {
      // Not speaking, maintain pre-speech buffer
      _addToPreSpeechBuffer(data);
    }
  }

  void _handleIntermediateFrame(Float32List data) {
    if (_speaking) {
      _speechBuffer.add(data);
      _redemptionCounter = 0;
    } else {
      _addToPreSpeechBuffer(data);
    }
  }

  @override
  void forceEndSpeech() {
    if (_speaking && _speechPositiveFrameCount >= _minSpeechFrames) {
      if (_isDebug) print('VAD Iterator: Forcing speech end.');
      _onVadEvent?.call(VadEvent(
        type: VadEventType.end,
        timestamp: _getCurrentTimestamp(),
        message:
            'Speech forcefully ended at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
        audioData: _combineSpeechBuffer(),
      ));
      // Reset state
      _speaking = false;
      _redemptionCounter = 0;
      _speechPositiveFrameCount = 0;
      _speechBuffer.clear();
      _preSpeechBuffer.clear();
      _speechStartIndex = 0;
    }
  }

  void _addToPreSpeechBuffer(Float32List data) {
    _preSpeechBuffer.add(data);
    while (_preSpeechBuffer.length > _preSpeechPadFrames) {
      _preSpeechBuffer.removeAt(0);
    }
  }

  double _getCurrentTimestamp() {
    return _currentSample / _sampleRate;
  }

  Uint8List _combineSpeechBuffer() {
    final int totalLength =
        _speechBuffer.fold(0, (sum, frame) => sum + frame.length);
    final Float32List combined = Float32List(totalLength);
    int offset = 0;
    for (var frame in _speechBuffer) {
      combined.setRange(offset, offset + frame.length, frame);
      offset += frame.length;
    }
    final int16Data = Int16List.fromList(
        combined.map((e) => (e * 32767).clamp(-32768, 32767).toInt()).toList());
    final Uint8List audioData = Uint8List.view(int16Data.buffer);
    return audioData;
  }

  Uint8List _combineSpeechBufferList(List<Float32List> frames) {
    final int totalLength = frames.fold(0, (sum, frame) => sum + frame.length);
    final Float32List combined = Float32List(totalLength);
    int offset = 0;
    for (var frame in frames) {
      combined.setRange(offset, offset + frame.length, frame);
      offset += frame.length;
    }
    final int16Data = Int16List.fromList(
        combined.map((e) => (e * 32767).clamp(-32768, 32767).toInt()).toList());
    final Uint8List audioData = Uint8List.view(int16Data.buffer);
    return audioData;
  }

  List<double> _convertBytesToFloat32(Uint8List data) {
    final buffer = data.buffer;
    final int16List = Int16List.view(buffer);
    return int16List.map((e) => e / 32768.0).toList();
  }
}

/// Create a VAD iterator for the non-web platform
/// [isDebug] - Whether to enable debug logging
/// [sampleRate] - Sample rate of the audio stream
/// [frameSamples] - Frame size in samples
/// [positiveSpeechThreshold] - Positive speech threshold
/// [negativeSpeechThreshold] - Negative speech threshold
/// [redemptionFrames] - Number of frames to wait before considering speech as valid
/// [preSpeechPadFrames] - Number of frames to pad before speech is considered valid
/// [minSpeechFrames] - Minimum number of speech frames required to consider speech as valid
/// [model] - Silero model version: 'legacy' (v4) or 'v5'
/// [baseAssetPath] - Base URL or path for model assets
/// [onnxWASMBasePath] - Base URL for ONNX Runtime WASM files (unused on non-web platforms)
/// [endSpeechPadFrames] - Number of frames to pad after speech ends
/// [numFramesToEmit] - Number of frames to accumulate before emitting chunk
Future<VadIterator> createVadIterator({
  required bool isDebug,
  required int sampleRate,
  required int frameSamples,
  required double positiveSpeechThreshold,
  required double negativeSpeechThreshold,
  required int redemptionFrames,
  required int preSpeechPadFrames,
  required int minSpeechFrames,
  required String model,
  required String baseAssetPath,
  required String onnxWASMBasePath,
  int endSpeechPadFrames = 1,
  int numFramesToEmit = 0,
}) {
  return VadIteratorNonWeb.create(
    isDebug: isDebug,
    sampleRate: sampleRate,
    frameSamples: frameSamples,
    positiveSpeechThreshold: positiveSpeechThreshold,
    negativeSpeechThreshold: negativeSpeechThreshold,
    redemptionFrames: redemptionFrames,
    preSpeechPadFrames: preSpeechPadFrames,
    minSpeechFrames: minSpeechFrames,
    model: model,
    baseAssetPath: baseAssetPath,
    onnxWASMBasePath: onnxWASMBasePath,
    endSpeechPadFrames: endSpeechPadFrames,
    numFramesToEmit: numFramesToEmit,
  );
}
