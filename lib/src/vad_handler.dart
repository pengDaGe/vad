// lib/src/vad_handler.dart

// Dart imports:
import 'dart:async';

// Package imports:
import 'package:record/record.dart';

import 'vad_handler_web.dart' if (dart.library.io) 'vad_handler_non_web.dart'
    as implementation;

/// Platform-agnostic Voice Activity Detection handler for real-time audio processing
///
/// Provides cross-platform VAD capabilities using Silero models with platform-specific
/// implementations for Web (AudioWorklet + ONNX Runtime JS) and native platforms
/// (AudioRecorder + ONNX Runtime Flutter).
abstract class VadHandler {
  /// Stream of speech end events containing processed audio data as floating point samples
  Stream<List<double>> get onSpeechEnd;

  /// Stream of frame processing events with speech probabilities and raw frame data
  Stream<({double isSpeech, double notSpeech, List<double> frame})>
      get onFrameProcessed;

  /// Stream of initial speech start detection events
  Stream<void> get onSpeechStart;

  /// Stream of validated speech start events (after minimum frame requirement)
  Stream<void> get onRealSpeechStart;

  /// Stream of VAD misfire events (false positive detections)
  Stream<void> get onVADMisfire;

  /// Stream of error events with descriptive error messages
  Stream<String> get onError;

  /// Start or resume listening for speech events with configurable parameters
  ///
  /// Default values are optimized for the legacy (v4) model. When using model='v5',
  /// the following parameters are automatically adjusted if not explicitly set:
  /// - preSpeechPadFrames: 1 → 3
  /// - redemptionFrames: 8 → 24
  /// - frameSamples: 1536 → 512
  /// - minSpeechFrames: 3 → 9
  ///
  /// [positiveSpeechThreshold] - Probability threshold for speech detection (0.0-1.0), default: 0.5
  /// [negativeSpeechThreshold] - Probability threshold for speech end detection (0.0-1.0), default: 0.35
  /// [preSpeechPadFrames] - Number of frames to include before speech starts, default: 1 (v4) or 3 (v5)
  /// [redemptionFrames] - Number of negative frames before ending speech, default: 8 (v4) or 24 (v5)
  /// [frameSamples] - Audio frame size in samples (512, 1024, or 1536 recommended), default: 1536 (v4) or 512 (v5)
  /// [minSpeechFrames] - Minimum frames required for valid speech detection, default: 3 (v4) or 9 (v5)
  /// [submitUserSpeechOnPause] - Whether to emit speech end event on pause, default: false
  /// [model] - VAD model version ('legacy' for v4, 'v5' for latest), default: 'legacy'
  /// [baseAssetPath] - Base URL or path for model assets, default: 'https://cdn.jsdelivr.net/npm/@keyurmaru/vad@0.0.1/'
  /// [onnxWASMBasePath] - Base URL for ONNX Runtime WASM files (Web only), default: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/'
  /// [recordConfig] - Custom audio recording configuration (native platforms only)
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
      RecordConfig? recordConfig});

  /// Stop listening and clean up audio resources
  Future<void> stopListening();

  /// Pause listening while maintaining audio stream (can be resumed)
  Future<void> pauseListening();

  /// Release all resources and close streams
  Future<void> dispose();

  /// Factory method to create platform-appropriate VAD handler instance
  ///
  /// [isDebug] - Enable debug logging for troubleshooting (default: false)
  ///
  /// Returns Web implementation on web platforms, native implementation otherwise.
  /// Supports Silero VAD models v4 ('legacy') and v5.
  static VadHandler create({bool isDebug = false}) {
    return implementation.createVadHandler(isDebug: isDebug);
  }
}
