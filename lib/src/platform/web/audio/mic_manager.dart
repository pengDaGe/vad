// lib/src/platform/web/audio/mic_manager.dart
//
// This file handles microphone access and management for web platform.

// ignore_for_file: public_member_api_docs, avoid_print

// Dart imports:
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

// Package imports:
import 'package:web/web.dart' as web;

// Project imports:
import 'package:vad/src/platform/web/audio/audio_capture.dart';

/// Manages microphone access and audio context for web platform.
/// This class handles:
/// - Requesting microphone permissions
/// - Creating and managing the audio context
/// - Connecting the microphone to the audio processing pipeline
class MicVAD {
  final web.AudioContext _audioContext;
  final web.MediaStream _stream;
  final AudioNodeVAD _audioNodeVAD;
  final web.MediaStreamAudioSourceNode _sourceNode;
  bool _listening = false;

  MicVAD._(
    this._audioContext,
    this._stream,
    this._audioNodeVAD,
    this._sourceNode,
  );

  /// Creates a new MicVAD instance with the specified options.
  /// This will request microphone permissions from the user.
  static Future<MicVAD> create(AudioNodeOptions options) async {
    try {
      if (options.isDebug) {
        print('MicVAD.create: Starting microphone initialization');
      }

      // Get microphone stream
      final constraints = {
        'audio': {
          'channelCount': 1,
          'echoCancellation': true,
          'autoGainControl': true,
          'noiseSuppression': true,
        },
      }.jsify()! as web.MediaStreamConstraints;

      final stream = await web.window.navigator.mediaDevices
          .getUserMedia(constraints)
          .toDart;

      if (options.isDebug) {
        print('MicVAD.create: Got media stream, creating audio context');
      }

      final audioContext = web.AudioContext();
      final sourceNode = audioContext.createMediaStreamSource(stream);
      final audioNodeVAD = await AudioNodeVAD.create(audioContext, options);
      audioNodeVAD.connect(sourceNode);

      return MicVAD._(audioContext, stream, audioNodeVAD, sourceNode);
    } catch (e, stackTrace) {
      print('Error creating MicVAD: $e');
      print('Stack trace: $stackTrace');
      rethrow;
    }
  }

  /// Pauses audio capture without releasing resources
  void pause() {
    _audioNodeVAD.pause();
    _listening = false;
  }

  /// Starts or resumes audio capture
  void start() {
    _audioNodeVAD.start();
    _listening = true;
  }

  /// Stops audio capture and releases all resources
  void destroy() {
    if (_listening) {
      pause();
    }

    // Stop all media stream tracks
    final streamJS = _stream as JSObject;
    final getTracks = streamJS.getProperty<JSFunction>('getTracks'.toJS);
    final tracksJS = getTracks.callAsFunction(streamJS) as JSArray;

    final length = tracksJS.length;
    for (int i = 0; i < length; i++) {
      final track = tracksJS.getProperty<JSObject>(i.toJS);
      final stopMethod = track.getProperty<JSFunction>('stop'.toJS);
      stopMethod.callAsFunction(track);
    }

    _sourceNode.disconnect();
    _audioNodeVAD.destroy();
    _audioContext.close();
  }

  /// Whether the MicVAD is currently listening
  bool get isListening => _listening;
}