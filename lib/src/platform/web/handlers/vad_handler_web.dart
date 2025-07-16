// lib/src/vad_handler_web.dart

// ignore_for_file: avoid_print

// Dart imports:
import 'dart:async';
import 'dart:typed_data';

// Package imports:
import 'package:record/record.dart';

// Project imports:
import 'package:vad/src/core/vad_event.dart';
import 'package:vad/src/core/vad_handler.dart';
import 'package:vad/src/platform/web/audio/audio_capture.dart';
import 'package:vad/src/platform/web/audio/mic_manager.dart';
import 'package:vad/src/platform/web/iterators/vad_iterator_web.dart';

/// Web platform implementation of VAD handler using AudioWorklet and ONNX Runtime JS
class VadHandlerWeb implements VadHandler {
  final StreamController<List<double>> _onSpeechEndController =
      StreamController<List<double>>.broadcast();
  final StreamController<
          ({double isSpeech, double notSpeech, List<double> frame})>
      _onFrameProcessedController = StreamController<
          ({
            double isSpeech,
            double notSpeech,
            List<double> frame
          })>.broadcast();
  final StreamController<void> _onSpeechStartController =
      StreamController<void>.broadcast();
  final StreamController<void> _onRealSpeechStartController =
      StreamController<void>.broadcast();
  final StreamController<void> _onVADMisfireController =
      StreamController<void>.broadcast();
  final StreamController<String> _onErrorController =
      StreamController<String>.broadcast();
  final StreamController<List<double>> _onEmitChunkController =
      StreamController<List<double>>.broadcast();

  bool _isDebug = false;
  MicVAD? _micVAD;
  VadIteratorWeb? _vadIterator;
  bool _isPaused = false;
  bool _submitUserSpeechOnPause = false;
  
  // Track current model parameters to detect changes
  String? _currentModel;
  int? _currentFrameSamples;
  double? _currentPositiveSpeechThreshold;
  double? _currentNegativeSpeechThreshold;
  int? _currentRedemptionFrames;
  int? _currentPreSpeechPadFrames;
  int? _currentMinSpeechFrames;
  String? _currentBaseAssetPath;
  String? _currentOnnxWASMBasePath;
  int? _currentEndSpeechPadFrames;
  int? _currentNumFramesToEmit;

  /// Constructor
  /// [isDebug] - Whether to enable debug logging (default: false)
  VadHandlerWeb._({bool isDebug = false}) {
    _isDebug = isDebug;
  }

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

  @override
  Future<void> startListening(
      {double positiveSpeechThreshold = 0.5,
      double negativeSpeechThreshold = 0.35,
      int preSpeechPadFrames = 1,
      int redemptionFrames = 8,
      int frameSamples = 1536,
      int minSpeechFrames = 3,
      bool submitUserSpeechOnPause = false,
      String model = 'v4',
      String baseAssetPath =
          'https://cdn.jsdelivr.net/npm/@keyurmaru/vad@0.0.1/',
      String onnxWASMBasePath =
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/',
      RecordConfig? recordConfig,
      int endSpeechPadFrames = 1,
      int numFramesToEmit = 0}) async {
    _submitUserSpeechOnPause = submitUserSpeechOnPause;

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

    if (_isDebug) {
      print('VadHandlerWeb: startListening: Creating VAD with parameters: '
          'positiveSpeechThreshold: $positiveSpeechThreshold, '
          'negativeSpeechThreshold: $negativeSpeechThreshold, '
          'preSpeechPadFrames: $preSpeechPadFrames, '
          'redemptionFrames: $redemptionFrames, '
          'frameSamples: $frameSamples, '
          'minSpeechFrames: $minSpeechFrames, '
          'submitUserSpeechOnPause: $submitUserSpeechOnPause, '
          'model: $model, '
          'baseAssetPath: $baseAssetPath, '
          'onnxWASMBasePath: $onnxWASMBasePath, '
          'endSpeechPadFrames: $endSpeechPadFrames, '
          'numFramesToEmit: $numFramesToEmit');
    }

    try {
      // Check if model parameters have changed
      final parametersChanged = _currentModel != model ||
          _currentFrameSamples != frameSamples ||
          _currentPositiveSpeechThreshold != positiveSpeechThreshold ||
          _currentNegativeSpeechThreshold != negativeSpeechThreshold ||
          _currentRedemptionFrames != redemptionFrames ||
          _currentPreSpeechPadFrames != preSpeechPadFrames ||
          _currentMinSpeechFrames != minSpeechFrames ||
          _currentBaseAssetPath != baseAssetPath ||
          _currentOnnxWASMBasePath != onnxWASMBasePath ||
          _currentEndSpeechPadFrames != endSpeechPadFrames ||
          _currentNumFramesToEmit != numFramesToEmit;
      
      // Check if we already have a MicVAD instance (resuming from pause)
      if (_micVAD != null && _isPaused && !parametersChanged) {
        if (_isDebug) {
          print('VadHandlerWeb: Resuming existing VAD instance from pause');
        }
        _vadIterator?.reset();
        _micVAD!.start();
        _isPaused = false;
      } else if (_micVAD == null || parametersChanged) {
        // Release old instances if parameters changed
        if (parametersChanged) {
          if (_isDebug) {
            print('VadHandlerWeb: Model parameters changed, recreating VAD');
          }
          _micVAD?.destroy();
          _micVAD = null;
          await _vadIterator?.release();
          _vadIterator = null;
        }
        if (_isDebug) {
          print('VadHandlerWeb: Creating AudioNodeVadOptions');
        }

        // Create VAD iterator
        _vadIterator = await VadIteratorWeb.create(
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

        // Set the callback
        _vadIterator!.setVadEventCallback(_handleVadEvent);

        // Create audio capture options
        final audioOptions = AudioNodeOptions(
          frameSamples: frameSamples,
          isDebug: _isDebug,
          onAudioFrame: (frame) {
            // Process audio frame through VAD iterator
            _vadIterator?.processFrame(frame);
          },
          onError: (error) {
            _onErrorController.add(error);
          },
        );

        if (_isDebug) {
          print('VadHandlerWeb: Creating MicVAD');
        }

        _micVAD = await MicVAD.create(audioOptions);
        _micVAD!.start();

        _isPaused = false;
        
        // Update current parameters
        _currentModel = model;
        _currentFrameSamples = frameSamples;
        _currentPositiveSpeechThreshold = positiveSpeechThreshold;
        _currentNegativeSpeechThreshold = negativeSpeechThreshold;
        _currentRedemptionFrames = redemptionFrames;
        _currentPreSpeechPadFrames = preSpeechPadFrames;
        _currentMinSpeechFrames = minSpeechFrames;
        _currentBaseAssetPath = baseAssetPath;
        _currentOnnxWASMBasePath = onnxWASMBasePath;
        _currentEndSpeechPadFrames = endSpeechPadFrames;
        _currentNumFramesToEmit = numFramesToEmit;

        if (_isDebug) {
          print('VadHandlerWeb: VAD created and started successfully');
        }
      } else {
        if (_isDebug) {
          print('VadHandlerWeb: VAD already running, ignoring start request');
        }
      }
    } catch (error, stackTrace) {
      print('VadHandlerWeb: Error starting VAD: $error');
      print('Stack trace: $stackTrace');
      _onErrorController.add(error.toString());
    }
  }

  void _handleVadEvent(VadEvent event) {
    switch (event.type) {
      case VadEventType.frameProcessed:
        if (event.probabilities != null && event.frameData != null) {
          if (_isDebug) {
            print(
                'VadHandlerWeb: onFrameProcessed: isSpeech: ${event.probabilities!.isSpeech}, notSpeech: ${event.probabilities!.notSpeech}');
          }
          _onFrameProcessedController.add((
            isSpeech: event.probabilities!.isSpeech,
            notSpeech: event.probabilities!.notSpeech,
            frame: event.frameData!,
          ));
        }
        break;
      case VadEventType.misfire:
        if (_isDebug) {
          print('VadHandlerWeb: onVADMisfire');
        }
        _onVADMisfireController.add(null);
        break;
      case VadEventType.start:
        if (_isDebug) {
          print('VadHandlerWeb: onSpeechStart');
        }
        _onSpeechStartController.add(null);
        break;
      case VadEventType.end:
        if (_isDebug) {
          print('VadHandlerWeb: onSpeechEnd');
        }
        if (event.audioData != null) {
          final int16Data = Int16List.view(event.audioData!.buffer);
          final floatData = int16Data.map((e) => e / 32768.0).toList();
          _onSpeechEndController.add(floatData);
        }
        break;
      case VadEventType.realStart:
        if (_isDebug) {
          print('VadHandlerWeb: onSpeechRealStart');
        }
        _onRealSpeechStartController.add(null);
        break;
      case VadEventType.error:
        if (_isDebug) {
          print('VadHandlerWeb: VAD error: ${event.message}');
        }
        _onErrorController.add(event.message);
        break;
      case VadEventType.chunk:
        if (_isDebug) {
          print('VadHandlerWeb: onEmitChunk');
        }
        if (event.audioData != null) {
          final int16Data = Int16List.view(event.audioData!.buffer);
          final floatData = int16Data.map((e) => e / 32768.0).toList();
          _onEmitChunkController.add(floatData);
        }
        break;
    }
  }

  @override
  Future<void> dispose() async {
    if (_isDebug) {
      print('VadHandlerWeb: dispose');
    }

    _micVAD?.destroy();
    _micVAD = null;
    await _vadIterator?.release();
    _vadIterator = null;
    _isPaused = false;
    
    // Reset current parameters
    _currentModel = null;
    _currentFrameSamples = null;
    _currentPositiveSpeechThreshold = null;
    _currentNegativeSpeechThreshold = null;
    _currentRedemptionFrames = null;
    _currentPreSpeechPadFrames = null;
    _currentMinSpeechFrames = null;
    _currentBaseAssetPath = null;
    _currentOnnxWASMBasePath = null;
    _currentEndSpeechPadFrames = null;
    _currentNumFramesToEmit = null;

    _onSpeechEndController.close();
    _onFrameProcessedController.close();
    _onSpeechStartController.close();
    _onRealSpeechStartController.close();
    _onVADMisfireController.close();
    _onErrorController.close();
    _onEmitChunkController.close();
  }

  @override
  Future<void> stopListening() async {
    if (_isDebug) {
      print('VadHandlerWeb: stopListening');
    }

    _micVAD?.destroy();
    _micVAD = null;
    await _vadIterator?.release();
    _vadIterator = null;
    _isPaused = false;
    
    // Reset current parameters
    _currentModel = null;
    _currentFrameSamples = null;
    _currentPositiveSpeechThreshold = null;
    _currentNegativeSpeechThreshold = null;
    _currentRedemptionFrames = null;
    _currentPreSpeechPadFrames = null;
    _currentMinSpeechFrames = null;
    _currentBaseAssetPath = null;
    _currentOnnxWASMBasePath = null;
    _currentEndSpeechPadFrames = null;
    _currentNumFramesToEmit = null;
  }

  @override
  Future<void> pauseListening() async {
    if (_isDebug) {
      print('VadHandlerWeb: pauseListening');
    }

    if (_micVAD != null) {
      _micVAD!.pause();
      _isPaused = true;

      if (_submitUserSpeechOnPause) {
        _vadIterator?.forceEndSpeech();
      }
    } else {
      if (_isDebug) {
        print('VadHandlerWeb: Cannot pause - no active VAD instance');
      }
    }
  }
}

/// Create a VAD handler for the web
/// [isDebug] - Enable debug logging for troubleshooting (default: false)
VadHandler createVadHandler({bool isDebug = false}) =>
    VadHandlerWeb._(isDebug: isDebug);
