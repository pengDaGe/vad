// lib/src/platform/web/audio/audio_capture.dart
//
// This file implements audio capture and preprocessing for web platform.
// It attempts to use AudioWorklet API for better performance, but falls back
// to ScriptProcessorNode if AudioWorklet is not available.

// ignore_for_file: public_member_api_docs, avoid_print

// Dart imports:
import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

// Package imports:
import 'package:web/web.dart' as web;

// Project imports:
import 'package:vad/src/platform/web/audio/audio_worklet_processor.dart';
import 'package:vad/src/platform/web/audio/resampler.dart';

/// Options for configuring the audio capture node
class AudioNodeOptions {
  final int frameSamples;
  final bool isDebug;
  final Function(Float32List frame)? onAudioFrame;
  final Function(String error)? onError;

  const AudioNodeOptions({
    this.frameSamples = 1536,
    this.isDebug = false,
    this.onAudioFrame,
    this.onError,
  });
}

/// Audio capture node that handles browser audio input and preprocessing.
/// This class is responsible for:
/// - Setting up AudioWorklet or ScriptProcessorNode
/// - Capturing audio from the browser
/// - Resampling to 16kHz
/// - Emitting audio frames via callbacks
class AudioNodeVAD {
  final web.AudioContext _context;
  final AudioNodeOptions _options;
  late final Resampler _resampler;

  web.AudioWorkletNode? _workletNode;
  web.ScriptProcessorNode? _scriptProcessorNode;
  web.GainNode? _gainNode;
  bool _active = false;
  int _processedFrames = 0;
  String? _processorBlobUrl;
  bool _useAudioWorklet = true;
  int _frameCount = 0;
  final List<double> _inputBuffer = [];

  AudioNodeVAD._(this._context, this._options);

  static Future<AudioNodeVAD> create(
    web.AudioContext context,
    AudioNodeOptions options,
  ) async {
    final instance = AudioNodeVAD._(context, options);
    await instance._initialize();
    return instance;
  }

  Future<void> _initialize() async {
    if (_options.isDebug) {
      print('AudioNodeVAD._initialize: Starting initialization');
    }

    // Create resampler for audio processing
    if (_options.isDebug) {
      print(
          'AudioNodeVAD: Creating resampler with nativeSampleRate: ${_context.sampleRate.toInt()}, targetSampleRate: 16000');
    }
    _resampler = Resampler(ResamplerOptions(
      nativeSampleRate: _context.sampleRate.toInt(),
      targetSampleRate: 16000, // Standard rate for audio processing
      targetFrameSize: _options.frameSamples,
    ));

    // Check if AudioWorklet is available using js_interop_unsafe
    bool audioWorkletAvailable = false;
    try {
      // Use js_interop_unsafe to check if the property exists
      final contextJS = _context as JSObject;
      audioWorkletAvailable = contextJS.has('audioWorklet');

      if (_options.isDebug) {
        print('AudioNodeVAD: AudioWorklet available: $audioWorkletAvailable');
      }
    } catch (e) {
      if (_options.isDebug) {
        print('AudioNodeVAD: Error checking for AudioWorklet availability: $e');
      }
      audioWorkletAvailable = false;
    }

    if (audioWorkletAvailable) {
      // Try to setup audio processing with AudioWorklet first
      try {
        await _setupAudioWorklet();
        _useAudioWorklet = true;
        if (_options.isDebug) {
          print('AudioNodeVAD: Using AudioWorklet for audio processing');
        }
      } catch (e) {
        if (_options.isDebug) {
          print(
              'AudioNodeVAD: AudioWorklet setup failed, falling back to ScriptProcessorNode: $e');
        }
        // Fall back to ScriptProcessorNode
        _useAudioWorklet = false;
        await _setupScriptProcessor();
        if (_options.isDebug) {
          print('AudioNodeVAD: Using ScriptProcessorNode for audio processing');
        }
      }
    } else {
      // AudioWorklet not available, use ScriptProcessorNode directly
      if (_options.isDebug) {
        print(
            'AudioNodeVAD: AudioWorklet not available, using ScriptProcessorNode for audio processing');
      }
      _useAudioWorklet = false;
      await _setupScriptProcessor();
    }
  }

  Future<void> _setupAudioWorklet() async {
    // Get audioWorklet reference using js_interop_unsafe
    final contextJS = _context as JSObject;
    final audioWorklet = contextJS['audioWorklet']!;

    // Generate processor code
    final processorCode = generateVadProcessorCode();

    // Create blob URL for the processor code
    final blob = web.Blob(
      [processorCode.toJS].toJS,
      web.BlobPropertyBag(type: 'application/javascript'),
    );
    _processorBlobUrl = web.URL.createObjectURL(blob);

    // Add the module to the audio worklet using js_interop_unsafe
    final promise = (audioWorklet as JSObject)
        .callMethod('addModule'.toJS, _processorBlobUrl!.toJS);
    await (promise as JSPromise).toDart;

    // Create AudioWorkletNode with options
    final processorOptions = {
      'bufferSize': 4096,
      'targetSampleRate': 16000,
      'frameSamples': _options.frameSamples,
      'isDebug': _options.isDebug,
    }.jsify()! as JSObject;

    final nodeOptions = web.AudioWorkletNodeOptions(
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1].map((e) => e.toJS).toList().toJS,
      processorOptions: processorOptions,
    );

    _workletNode = web.AudioWorkletNode(
      _context,
      'vad-audio-processor',
      nodeOptions,
    );

    // Create gain node with zero gain to handle the audio chain
    _gainNode = _context.createGain();
    _gainNode!.gain.value = 0.0;

    // Set up message port communication
    _setupMessagePort();

    // Set up processor error handler
    _setupProcessorErrorHandler();

    // Connect audio chain
    _workletNode!.connect(_gainNode!);
    _gainNode!.connect(_context.destination);

    if (_options.isDebug) {
      print(
          'AudioNodeVAD: Audio setup complete. Context state: ${_context.state}');
    }
  }

  Future<void> _setupScriptProcessor() async {
    try {
      // Create ScriptProcessorNode
      // Using 4096 buffer size for good balance between latency and performance
      const bufferSize = 4096;

      // Create the ScriptProcessorNode using js_interop_unsafe
      final contextJS = _context as JSObject;
      _scriptProcessorNode = contextJS.callMethod(
        'createScriptProcessor'.toJS,
        bufferSize.toJS,
        1.toJS, // input channels
        1.toJS, // output channels
      ) as web.ScriptProcessorNode;

      // Create gain node with zero gain to handle the audio chain
      _gainNode = _context.createGain();
      _gainNode!.gain.value = 0.0;

      // Set up the audio processing event handler
      _setupScriptProcessorHandler();

      // Connect audio chain
      _scriptProcessorNode!.connect(_gainNode!);
      _gainNode!.connect(_context.destination);

      if (_options.isDebug) {
        print(
            'AudioNodeVAD: ScriptProcessor setup complete. Buffer size: $bufferSize, Context state: ${_context.state}');
      }
    } catch (e) {
      print('AudioNodeVAD: Error setting up ScriptProcessorNode: $e');
      _options.onError?.call('Failed to setup ScriptProcessorNode: $e');
      rethrow;
    }
  }

  void _setupScriptProcessorHandler() {
    final processorJS = _scriptProcessorNode as JSObject;
    processorJS['onaudioprocess'] = ((web.AudioProcessingEvent event) {
      if (!_active) return;

      try {
        // Get input buffer
        final inputBuffer = event.inputBuffer;
        final inputData = inputBuffer.getChannelData(0);

        final float32List = inputData.toDart;

        // Add samples to our buffer
        for (var i = 0; i < float32List.length; i++) {
          _inputBuffer.add(float32List[i]);
        }

        // Process complete frames
        const bufferSize = 4096;
        while (_inputBuffer.length >= bufferSize) {
          final frame =
              Float32List.fromList(_inputBuffer.take(bufferSize).toList());
          _inputBuffer.removeRange(0, bufferSize);

          _frameCount++;
          _processedFrames++;

          // Process through resampler and emit frames
          final frames = _resampler.process(frame);
          for (final resampledFrame in frames) {
            _options.onAudioFrame?.call(resampledFrame);
          }

          if (_options.isDebug && _frameCount % 100 == 0) {
            print(
                'AudioNodeVAD: ScriptProcessor processed $_frameCount frames');
          }
        }
      } catch (e) {
        print('AudioNodeVAD: Error in ScriptProcessor audio processing: $e');
        _options.onError?.call('ScriptProcessor audio processing error: $e');
      }
    }).toJS;
  }

  void _setupMessagePort() {
    _workletNode!.port.onmessage = ((web.MessageEvent event) {
      final jsData = event.data as JSObject;
      final data = jsData.dartify() as Map<Object?, Object?>;

      final type = data['type'] as String?;

      switch (type) {
        case 'ready':
          if (_options.isDebug) {
            print('AudioNodeVAD: Processor ready');
          }
          break;

        case 'audioFrame':
          _handleAudioFrame(data);
          break;
      }
    }).toJS;
  }

  void _setupProcessorErrorHandler() {
    // Listen for processor errors
    _workletNode!.onprocessorerror = ((web.Event event) {
      print('AudioNodeVAD: CRITICAL - AudioWorkletProcessor error occurred');

      // The processor will output silence after an error, so we need to notify the consumer
      _options.onError?.call(
          'AudioWorkletProcessor error: The processor encountered an error and will output silence. Please restart audio capture.');

      // Log additional debug info if available
      if (_options.isDebug) {
        print('AudioNodeVAD: Processor error details:');
        print('  - Event type: ${event.type}');
        print('  - Processed frames before error: $_processedFrames');
        print('  - Active state: $_active');
      }

      // Since the processor will output silence, we should mark it as inactive
      _active = false;
    }).toJS;
  }

  void _handleAudioFrame(Map<Object?, Object?> data) {
    if (!_active) return;

    try {
      final frameData = (data['frame'] as List).cast<double>();
      final frame = Float32List.fromList(frameData);

      _processedFrames++;

      // Process through resampler and emit frames
      final frames = _resampler.process(frame);
      for (final resampledFrame in frames) {
        _options.onAudioFrame?.call(resampledFrame);
      }
    } catch (e) {
      print('AudioNodeVAD: Error processing audio frame: $e');
      _options.onError?.call('Audio frame processing error: $e');
    }
  }

  void pause() {
    if (_options.isDebug) {
      print('AudioNodeVAD: Pausing (processed $_processedFrames frames)');
    }
    _active = false;

    if (_useAudioWorklet && _workletNode != null) {
      _workletNode!.port.postMessage({'type': 'pause'}.jsify());
    }
  }

  void start() {
    if (_options.isDebug) {
      print('AudioNodeVAD: Starting');
    }
    _active = true;
    _processedFrames = 0;
    _frameCount = 0;
    _inputBuffer.clear();

    if (_useAudioWorklet && _workletNode != null) {
      _workletNode!.port.postMessage({'type': 'start'}.jsify());
    }
  }

  void connect(web.AudioNode node) {
    if (_useAudioWorklet && _workletNode != null) {
      node.connect(_workletNode!);
    } else if (!_useAudioWorklet && _scriptProcessorNode != null) {
      node.connect(_scriptProcessorNode!);
    }
  }

  void destroy() {
    if (_options.isDebug) {
      print(
          'AudioNodeVAD: Destroying (processed $_processedFrames frames total)');
    }

    _active = false;

    if (_useAudioWorklet && _workletNode != null) {
      _workletNode!.port.postMessage({'type': 'destroy'}.jsify());
      _workletNode!.disconnect();
    } else if (!_useAudioWorklet && _scriptProcessorNode != null) {
      _scriptProcessorNode!.disconnect();
      // Clear the event handler
      final processorJS = _scriptProcessorNode as JSObject;
      processorJS['onaudioprocess'] = null;
    }

    _gainNode?.disconnect();

    // Clean up blob URL if we used AudioWorklet
    if (_processorBlobUrl != null) {
      web.URL.revokeObjectURL(_processorBlobUrl!);
    }
  }
}