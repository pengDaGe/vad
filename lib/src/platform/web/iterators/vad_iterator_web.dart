// lib/src/vad_iterator_web.dart
// ignore_for_file: avoid_print

// Dart imports:
import 'dart:async';
import 'dart:typed_data';

// Project imports:
import 'package:vad/src/core/vad_iterator.dart';
import 'package:vad/src/core/vad_event.dart';
import 'package:vad/src/core/vad_model.dart';
import 'package:vad/src/platform/web/iterators/vad_inference_web.dart';
import 'package:vad/src/utils/model_utils.dart';

/// Web platform VAD iterator using ONNX Runtime JS for browser-based speech detection
class VadIteratorWeb implements VadIterator {
  /// Whether to enable debug logging
  final bool _isDebug;

  /// Positive speech threshold
  final double _positiveSpeechThreshold;

  /// Negative speech threshold
  final double _negativeSpeechThreshold;

  /// Number of frames to wait before considering speech as valid
  final int _redemptionFrames;

  /// Frame size in samples - WARNING: Use 512, 1024, or 1536 for optimal model performance
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

  /// Silero model version: 'v4' or 'v5'
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

  /// Number of frames processed
  int _totalFramesProcessed = 0;

  /// VAD model
  VadModel? _vadModel;

  /// Callback for VAD events
  VadEventCallback? _onVadEvent;

  /// Buffer for audio data
  final List<int> _byteBuffer = [];

  /// Number of bytes per frame
  final int _frameByteCount;

  /// Whether the real speech start event has been fired
  bool _speechRealStartFired = false;

  /// Index in speech buffer where current speech segment starts
  int _speechStartIndex = 0;

  /// Number of redemption frames that have been sent in chunks
  int _sentRedemptionFrames = 0;

  /// Constructor
  /// [isDebug] - Whether to enable debug logging
  /// [sampleRate] - Sample rate of the audio stream
  /// [frameSamples] - Frame size in samples
  /// [positiveSpeechThreshold] - Positive speech threshold
  /// [negativeSpeechThreshold] - Negative speech threshold
  /// [redemptionFrames] - Number of frames to wait before considering speech as valid
  /// [preSpeechPadFrames] - Number of frames to pad before speech is considered valid
  /// [minSpeechFrames] - Minimum number of speech frames required to consider speech as valid
  /// [model] - Silero model version: 'v4' or 'v5'
  /// [endSpeechPadFrames] - Number of frames to pad after speech ends
  /// [numFramesToEmit] - Number of frames to accumulate before emitting chunk
  VadIteratorWeb._internal({
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

  /// Create and initialize a VadIteratorWeb instance
  static Future<VadIteratorWeb> create({
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
  }) async {
    final instance = VadIteratorWeb._internal(
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
    await instance._initModelWithWasmPath(baseAssetPath, onnxWASMBasePath);

    return instance;
  }

  /// Initialize the VAD model with the given model path and ONNX WASM base path
  /// [baseAssetPath] - Base path to the model assets
  /// [onnxWASMBasePath] - Path to the ONNX WASM base path
  Future<void> _initModelWithWasmPath(
      String baseAssetPath, String onnxWASMBasePath) async {
    try {
      final modelUrl = getModelUrl(baseAssetPath, _model);

      if (_isDebug) {
        print('VadIteratorWeb: Initializing model with:');
        print('  - baseAssetPath: $baseAssetPath');
        print('  - model: $_model');
        print('  - computed modelUrl: $modelUrl');
        print('  - onnxWASMBasePath: $onnxWASMBasePath');
      }

      if (_model == 'v5') {
        _vadModel = await SileroV5Model.create(modelUrl, onnxWASMBasePath);
      } else {
        _vadModel = await SileroV4Model.create(modelUrl, onnxWASMBasePath);
      }

      if (_isDebug) print('VAD model initialized from $modelUrl.');
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
    if (_isDebug) {
      print(
          'VadIteratorWeb: Resetting state (processed $_totalFramesProcessed frames so far)');
    }
    _speaking = false;
    _redemptionCounter = 0;
    _speechPositiveFrameCount = 0;
    _currentSample = 0;
    _speechRealStartFired = false;
    _speechStartIndex = 0;
    _sentRedemptionFrames = 0;
    _preSpeechBuffer.clear();
    _speechBuffer.clear();
    _byteBuffer.clear();
    _totalFramesProcessed = 0;
  }

  @override
  Future<void> release() async {
    await _vadModel?.release();
    _vadModel = null;
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
      await processFrame(Float32List.fromList(frameData));
    }
  }

  /// Process a single frame of audio data
  /// [frame] - The frame of audio data to process
  Future<void> processFrame(Float32List frame) async {
    if (_vadModel == null) {
      print('VAD Iterator: Model not initialized.');
      return;
    }

    if (frame.length != _frameSamples) {
      print(
          'VADIteratorWeb: Unexpected frame size: ${frame.length}, expected: $_frameSamples');
      return;
    }

    _totalFramesProcessed++;

    try {
      final speechProb = await _runModelInference(frame);

      if (_speaking && speechProb < _negativeSpeechThreshold && _isDebug) {
        print(
            'VadIteratorWeb: During speech - probability ${speechProb.toStringAsFixed(3)} < negativeSpeechThreshold ${_negativeSpeechThreshold.toStringAsFixed(3)}');
      }

      final frameData = frame.toList();
      _onVadEvent?.call(VadEvent(
        type: VadEventType.frameProcessed,
        timestamp: _getCurrentTimestamp(),
        message:
            'Frame processed at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
        probabilities: SpeechProbabilities(
            isSpeech: speechProb, notSpeech: 1.0 - speechProb),
        frameData: frameData,
      ));

      _currentSample += _frameSamples;
      _handleStateTransitions(speechProb, frame);
    } catch (e, stackTrace) {
      print('VadIteratorWeb: Error in _processFrame: $e');
      print('Stack trace: $stackTrace');

      // Send error event
      _onVadEvent?.call(VadEvent(
        type: VadEventType.error,
        timestamp: _getCurrentTimestamp(),
        message: 'Frame processing error: $e',
      ));
    }
  }

  Future<double> _runModelInference(Float32List data) async {
    try {
      final probs = await _vadModel!.process(data);
      return probs.isSpeech;
    } catch (e) {
      print('VadIteratorWeb: Model inference error: $e');
      rethrow;
    }
  }

  void _handleStateTransitions(double speechProb, Float32List data) {
    if (speechProb >= _positiveSpeechThreshold) {
      if (!_speaking) {
        _speaking = true;
        _speechRealStartFired = false;
        _speechStartIndex = 0;
        if (_isDebug) {
          print(
              'VadIteratorWeb: Speech started (prob: ${speechProb.toStringAsFixed(3)})');
        }
        _onVadEvent?.call(VadEvent(
          type: VadEventType.start,
          timestamp: _getCurrentTimestamp(),
          message:
              'Speech started at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
        ));
        _speechBuffer.addAll(_preSpeechBuffer);
        _preSpeechBuffer.clear();
      }
      if (_redemptionCounter > 0) {
        if (_isDebug) {
          print(
              'VadIteratorWeb: Redemption counter reset from $_redemptionCounter to 0 due to positive speech (prob: ${speechProb.toStringAsFixed(3)})');
        }
      }
      _redemptionCounter = 0;
      _sentRedemptionFrames = 0;
      _speechBuffer.add(data);
      _speechPositiveFrameCount++;

      if (_speechPositiveFrameCount == _minSpeechFrames &&
          !_speechRealStartFired) {
        _speechRealStartFired = true;
        if (_isDebug) {
          print('VadIteratorWeb: Real speech validated');
        }
        _onVadEvent?.call(VadEvent(
          type: VadEventType.realStart,
          timestamp: _getCurrentTimestamp(),
          message:
              'Speech validated at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
        ));
      }
    } else if (speechProb < _negativeSpeechThreshold) {
      _handleSpeechNegativeFrame(data);
    } else {
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
      _sentRedemptionFrames = _redemptionCounter;
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
      _speechBuffer.add(data);
      _redemptionCounter++;
      if (_isDebug) {
        print(
            'VadIteratorWeb: Redemption counter incremented to $_redemptionCounter/$_redemptionFrames');
      }

      if (_redemptionCounter >= _redemptionFrames) {
        _speaking = false;
        _redemptionCounter = 0;

        if (_speechPositiveFrameCount >= _minSpeechFrames) {
          if (_isDebug) {
            print(
                'VadIteratorWeb: Speech ended (duration: ${(_speechBuffer.length * _frameSamples / _sampleRate).toStringAsFixed(2)}s)');
          }

          // Calculate the audio buffer with endSpeechPadFrames
          final frames = _speechBuffer;
          final audioEndIndex = frames.length - (_redemptionFrames - _endSpeechPadFrames);
          
          List<Float32List> audioBufferPad;
          if (_numFramesToEmit > 0) {
            // When chunks are enabled, only include frames from speechStartIndex onwards
            // This ensures onSpeechEnd only contains frames since the last onEmitChunk
            audioBufferPad = frames.sublist(_speechStartIndex, audioEndIndex);
          } else {
            // When chunks are disabled, include all frames
            audioBufferPad = frames.sublist(0, audioEndIndex);
          }
          final audio = _combineSpeechBufferList(audioBufferPad);

          _onVadEvent?.call(VadEvent(
            type: VadEventType.end,
            timestamp: _getCurrentTimestamp(),
            message:
                'Speech ended at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
            audioData: audio,
          ));

          // Handle final chunk emission - emit all accumulated frames since last onEmitChunk
          if (_sentRedemptionFrames == 0 && _numFramesToEmit > 0) {
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
          } else if (_sentRedemptionFrames > 0 &&
              _endSpeechPadFrames > _sentRedemptionFrames &&
              _numFramesToEmit > 0) {
            final speechEndIndex =
                _speechStartIndex + _endSpeechPadFrames - _sentRedemptionFrames;
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
          if (_isDebug) {
            print(
                'VadIteratorWeb: Misfire (only $_speechPositiveFrameCount positive frames)');
          }
          _onVadEvent?.call(VadEvent(
            type: VadEventType.misfire,
            timestamp: _getCurrentTimestamp(),
            message:
                'Misfire detected at ${_getCurrentTimestamp().toStringAsFixed(3)}s',
          ));
        }
        _speechPositiveFrameCount = 0;
        _speechRealStartFired = false;
        _speechStartIndex = 0;
        _sentRedemptionFrames = 0;

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
      }
    } else {
      _addToPreSpeechBuffer(data);
    }
  }

  void _handleIntermediateFrame(Float32List data) {
    if (_speaking) {
      _speechBuffer.add(data);
      _redemptionCounter = 0;
      // DO NOT reset _sentRedemptionFrames here - this is critical!
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
      _speaking = false;
      _redemptionCounter = 0;
      _speechPositiveFrameCount = 0;
      _speechBuffer.clear();
      _preSpeechBuffer.clear();
      _speechRealStartFired = false;
      _speechStartIndex = 0;
      _sentRedemptionFrames = 0;
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
        combined.map((e) => (e * 32768).clamp(-32768, 32767).toInt()).toList());
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
        combined.map((e) => (e * 32768).clamp(-32768, 32767).toInt()).toList());
    final Uint8List audioData = Uint8List.view(int16Data.buffer);
    return audioData;
  }

  List<double> _convertBytesToFloat32(Uint8List data) {
    final buffer = data.buffer;
    final int16List = Int16List.view(buffer);
    return int16List.map((e) => e / 32768.0).toList();
  }
}

/// Create a VAD iterator for the web platform
/// [isDebug] - Whether to enable debug logging
/// [sampleRate] - Sample rate of the audio stream
/// [frameSamples] - Frame size in samples
/// [positiveSpeechThreshold] - Positive speech threshold
/// [negativeSpeechThreshold] - Negative speech threshold
/// [redemptionFrames] - Number of frames to wait before considering speech as valid
/// [preSpeechPadFrames] - Number of frames to pad before speech is considered valid
/// [minSpeechFrames] - Minimum number of speech frames required to consider speech as valid
/// [model] - Silero model version: 'v4' or 'v5'
/// [baseAssetPath] - Base URL or path for model assets
/// [onnxWASMBasePath] - Base URL for ONNX Runtime WASM files (Web only)
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
  return VadIteratorWeb.create(
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
