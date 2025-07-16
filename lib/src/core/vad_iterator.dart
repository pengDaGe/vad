// lib/src/vad_iterator.dart

// Dart imports:
import 'dart:typed_data';

// Project imports:
import 'vad_event.dart';

import 'package:vad/src/platform/web/iterators/vad_iterator_web.dart'
    if (dart.library.io) 'package:vad/src/platform/native/iterators/vad_iterator_native.dart'
    as implementation;

/// Low-level Voice Activity Detection iterator for direct audio processing
///
/// Provides frame-by-frame VAD processing with platform-specific implementations.
/// Used internally but can be used directly for more granular
/// control over VAD processing, such as processing non-streaming audio data.
abstract class VadIterator {
  /// Reset VAD state and clear internal buffers
  void reset();

  /// Release model resources and cleanup memory
  Future<void> release();

  /// Set callback function to receive VAD events
  void setVadEventCallback(VadEventCallback callback);

  /// Process raw audio data bytes and emit VAD events
  Future<void> processAudioData(Uint8List data);

  /// Force speech end detection
  void forceEndSpeech();

  /// Factory method to create platform-appropriate VAD iterator
  ///
  /// [isDebug] - Enable debug logging
  /// [sampleRate] - Audio sample rate (typically 16000 Hz)
  /// [frameSamples] - Frame size in samples (512, 1024, or 1536 recommended)
  /// [positiveSpeechThreshold] - Threshold for detecting speech start (0.0-1.0)
  /// [negativeSpeechThreshold] - Threshold for detecting speech end (0.0-1.0)
  /// [redemptionFrames] - Frames of silence before ending speech
  /// [preSpeechPadFrames] - Frames to include before speech detection
  /// [minSpeechFrames] - Minimum frames required for valid speech
  /// [model] - Silero model version ('v4' or 'v5')
  /// [baseAssetPath] - Base URL or path for model assets
  /// [onnxWASMBasePath] - Base URL for ONNX Runtime WASM files (Web only)
  /// [endSpeechPadFrames] - Frames to append after speech detection
  /// [numFramesToEmit] - Number of frames before emitting chunk events
  static Future<VadIterator> create(
      {required bool isDebug,
      required int sampleRate,
      required int frameSamples,
      required double positiveSpeechThreshold,
      required double negativeSpeechThreshold,
      required int redemptionFrames,
      required int preSpeechPadFrames,
      required int minSpeechFrames,
      required String model,
      String baseAssetPath =
          'https://cdn.jsdelivr.net/npm/@keyurmaru/vad@0.0.1/',
      String onnxWASMBasePath =
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/',
      int endSpeechPadFrames = 1,
      int numFramesToEmit = 0}) {
    return implementation.createVadIterator(
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
        numFramesToEmit: numFramesToEmit);
  }
}

/// Callback function type for receiving VAD events during audio processing
typedef VadEventCallback = void Function(VadEvent event);
