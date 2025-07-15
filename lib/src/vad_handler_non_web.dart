// lib/src/vad_handler_non_web.dart

// ignore_for_file: avoid_print

// Dart imports:
import 'dart:async';

// Package imports:
import 'package:record/record.dart';

// Project imports:
import 'package:vad/src/vad_handler.dart';
import 'package:vad/src/vad_iterator.dart';
import 'vad_event.dart';

/// Native platform implementation of VAD handler using AudioRecorder and ONNX Runtime Flutter
class VadHandlerNonWeb implements VadHandler {
  final AudioRecorder _audioRecorder = AudioRecorder();
  late VadIterator _vadIterator;
  StreamSubscription<List<int>>? _audioStreamSubscription;

  bool _isDebug = false;
  bool _isInitialized = false;
  bool _submitUserSpeechOnPause = false;
  bool _isPaused = false;

  final _onSpeechEndController = StreamController<List<double>>.broadcast();
  final _onFrameProcessedController = StreamController<
      ({double isSpeech, double notSpeech, List<double> frame})>.broadcast();
  final _onSpeechStartController = StreamController<void>.broadcast();
  final _onRealSpeechStartController = StreamController<void>.broadcast();
  final _onVADMisfireController = StreamController<void>.broadcast();
  final _onErrorController = StreamController<String>.broadcast();
  final _onEmitChunkController = StreamController<List<double>>.broadcast();

  @override
  Stream<List<double>> get onSpeechEnd => _onSpeechEndController.stream;

  @override
  Stream<({double isSpeech, double notSpeech, List<double> frame})>
      get onFrameProcessed => _onFrameProcessedController.stream;

  @override
  Stream<void> get onSpeechStart => _onSpeechStartController.stream;

  @override
  Stream<void> get onRealSpeechStart => _onRealSpeechStartController.stream;

  @override
  Stream<void> get onVADMisfire => _onVADMisfireController.stream;

  @override
  Stream<String> get onError => _onErrorController.stream;

  @override
  Stream<List<double>> get onEmitChunk => _onEmitChunkController.stream;

  /// Constructor
  /// [isDebug] - Whether to enable debug logging (default: false)
  VadHandlerNonWeb._({bool isDebug = false}) {
    _isDebug = isDebug;
  }

  void _handleVadEvent(VadEvent event) {
    if (_isDebug) {
      print(
          'VadHandlerNonWeb: VAD Event: ${event.type} with message ${event.message}');
    }
    switch (event.type) {
      case VadEventType.start:
        _onSpeechStartController.add(null);
        break;
      case VadEventType.realStart:
        _onRealSpeechStartController.add(null);
        break;
      case VadEventType.end:
        if (event.audioData != null) {
          final int16List = event.audioData!.buffer.asInt16List();
          final floatSamples = int16List.map((e) => e / 32768.0).toList();
          _onSpeechEndController.add(floatSamples);
        }
        break;
      case VadEventType.frameProcessed:
        if (event.probabilities != null && event.frameData != null) {
          _onFrameProcessedController.add((
            isSpeech: event.probabilities!.isSpeech,
            notSpeech: event.probabilities!.notSpeech,
            frame: event.frameData!
          ));
        }
        break;
      case VadEventType.misfire:
        _onVADMisfireController.add(null);
        break;
      case VadEventType.error:
        _onErrorController.add(event.message);
        break;
      case VadEventType.chunk:
        if (event.audioData != null) {
          final int16List = event.audioData!.buffer.asInt16List();
          final floatSamples = int16List.map((e) => e / 32768.0).toList();
          _onEmitChunkController.add(floatSamples);
        }
        break;
    }
  }

  @override
  Future<void> startListening(
      {double positiveSpeechThreshold = 0.5,
      double negativeSpeechThreshold = 0.35,
      int preSpeechPadFrames = 1,
      int redemptionFrames = 8,
      int frameSamples = 1536,
      int minSpeechFrames = 3,
      bool submitUserSpeechOnPause = false,
      String model = 'legacy',
      String baseAssetPath =
          'https://cdn.jsdelivr.net/npm/@keyurmaru/vad@0.0.1/',
      String onnxWASMBasePath =
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/',
      RecordConfig? recordConfig,
      int endSpeechPadFrames = 1,
      int numFramesToEmit = 0}) async {
    // Adjust parameters for v5 model if using defaults
    if (model == 'v5') {
      if (preSpeechPadFrames == 1) {
        preSpeechPadFrames = 3;
      }
      if (redemptionFrames == 8) {
        redemptionFrames = 24;
      }
      if (frameSamples == 1536) {
        frameSamples = 512;
      }
      if (minSpeechFrames == 3) {
        minSpeechFrames = 9;
      }
      if (endSpeechPadFrames == 1) {
        endSpeechPadFrames = 3;
      }
    }

    if (_isPaused && _audioStreamSubscription != null) {
      if (_isDebug) print('VadHandlerNonWeb: Resuming from paused state');
      _isPaused = false;
      return;
    }

    if (!_isInitialized) {
      _vadIterator = await VadIterator.create(
        isDebug: _isDebug,
        sampleRate: 16000,
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
      _vadIterator.setVadEventCallback(_handleVadEvent);
      _submitUserSpeechOnPause = submitUserSpeechOnPause;
      _isInitialized = true;
    }

    bool hasPermission = await _audioRecorder.hasPermission();
    if (!hasPermission) {
      _onErrorController
          .add('VadHandlerNonWeb: No permission to record audio.');
      print('VadHandlerNonWeb: No permission to record audio.');
      return;
    }

    _isPaused = false;
    final config = recordConfig ??
        const RecordConfig(
            encoder: AudioEncoder.pcm16bits,
            sampleRate: 16000,
            bitRate: 16,
            numChannels: 1,
            echoCancel: true,
            autoGain: true,
            noiseSuppress: true);
    final stream = await _audioRecorder.startStream(config);

    _audioStreamSubscription = stream.listen((data) async {
      if (!_isPaused) {
        await _vadIterator.processAudioData(data);
      }
    });
  }

  @override
  Future<void> stopListening() async {
    if (_isDebug) print('stopListening');
    try {
      if (_submitUserSpeechOnPause) {
        _vadIterator.forceEndSpeech();
      }

      await _audioStreamSubscription?.cancel();
      _audioStreamSubscription = null;
      await _audioRecorder.stop();
      _vadIterator.reset();
      _isPaused = false;
    } catch (e) {
      _onErrorController.add(e.toString());
      print('Error stopping audio stream: $e');
    }
  }

  @override
  Future<void> pauseListening() async {
    if (_isDebug) print('pauseListening');
    _isPaused = true;
    if (_submitUserSpeechOnPause) {
      _vadIterator.forceEndSpeech();
    }
  }

  @override
  Future<void> dispose() async {
    if (_isDebug) print('VadHandlerNonWeb: dispose');
    await stopListening();
    await _audioRecorder.dispose();
    _audioStreamSubscription?.cancel();
    _audioStreamSubscription = null;
    _isInitialized = false;
    _vadIterator.release();
    _onSpeechEndController.close();
    _onFrameProcessedController.close();
    _onSpeechStartController.close();
    _onRealSpeechStartController.close();
    _onVADMisfireController.close();
    _onErrorController.close();
    _onEmitChunkController.close();
  }
}

/// Create a VAD handler for the non-web platforms
/// [isDebug] - Enable debug logging for troubleshooting (default: false)
VadHandler createVadHandler({bool isDebug = false}) =>
    VadHandlerNonWeb._(isDebug: isDebug);
