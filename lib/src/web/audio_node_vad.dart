// lib/src/web/audio_node_vad.dart
//
// This file implements audio processing for VAD on web platform.
// It attempts to use AudioWorklet API for better performance, but falls back
// to ScriptProcessorNode if AudioWorklet is not available (older browsers,
// non-secure contexts, or other compatibility issues).

// ignore_for_file: public_member_api_docs, avoid_print

// Dart imports:
import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

// Package imports:
import 'package:web/web.dart' as web;

// Project imports:
import 'package:vad/src/vad_event.dart';
import '../vad_iterator_web.dart';
import 'resampler.dart';

String generateVadProcessorCode() {
  return '''
class VadAudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    const opts = options.processorOptions || {};
    this.bufferSize = opts.bufferSize || 4096;
    this.targetSampleRate = opts.targetSampleRate || 16000;
    this.frameSamples = opts.frameSamples || 1536;
    this.isDebug = opts.isDebug || false;

    this.inputBuffer = [];
    this.outputBuffer = [];
    this.frameCount = 0;
    this.lastProcessTime = Date.now();
    this.active = false;

    this.port.onmessage = (event) => {
      this.handleMessage(event.data);
    };

    this.port.postMessage({ type: 'ready' });

    if (this.isDebug) {
      console.log('VadAudioProcessor: Initialized with options:', opts);
    }
  }

  handleMessage(data) {
    switch (data.type) {
      case 'start':
        this.active = true;
        this.frameCount = 0;
        if (this.isDebug) {
          console.log('VadAudioProcessor: Started');
        }
        break;

      case 'pause':
        this.active = false;
        if (this.isDebug) {
          console.log('VadAudioProcessor: Paused');
        }
        break;

      case 'destroy':
        this.active = false;
        this.inputBuffer = [];
        this.outputBuffer = [];
        if (this.isDebug) {
          console.log('VadAudioProcessor: Destroyed');
        }
        break;
    }
  }

  process(inputs, outputs, parameters) {
    try {
      const input = inputs[0];

      if (!input || input.length === 0) {
        return true; // Keep processor alive
      }

      const inputChannel = input[0];

      const output = outputs[0];
      if (output && output.length > 0) {
        output[0].fill(0);
      }

      if (!this.active) {
        return true; // Keep processor alive but don't process
      }

      for (let i = 0; i < inputChannel.length; i++) {
        this.inputBuffer.push(inputChannel[i]);
      }

      while (this.inputBuffer.length >= this.bufferSize) {
        const frame = this.inputBuffer.splice(0, this.bufferSize);
        this.frameCount++;

        this.port.postMessage({
          type: 'audioFrame',
          frame: frame,
          frameNumber: this.frameCount,
          timestamp: currentTime
        });

        if (this.isDebug && this.frameCount % 100 === 0) {
          const now = Date.now();
          const timeSinceLastLog = now - this.lastProcessTime;
          console.log(`VadAudioProcessor: Processed \${this.frameCount} frames, time since last log: \${timeSinceLastLog}ms`);
          this.lastProcessTime = now;
        }
      }

      return true; // Keep processor alive
    } catch (error) {
      console.error('VadAudioProcessor: Error in process method:', error);
      throw error;
    }
  }
}

registerProcessor('vad-audio-processor', VadAudioProcessor);
''';
}

class AudioNodeVadOptions {
  final double positiveSpeechThreshold;
  final double negativeSpeechThreshold;
  final int redemptionFrames;
  final int preSpeechPadFrames;
  final int endSpeechPadFrames;
  final int frameSamples;
  final int minSpeechFrames;
  final int numFramesToEmit;
  final bool submitUserSpeechOnPause;
  final void Function(VadEvent) onVadEvent;
  final String model;
  final String baseAssetPath;
  final String onnxWASMBasePath;
  final bool isDebug;

  const AudioNodeVadOptions({
    this.positiveSpeechThreshold = 0.5,
    this.negativeSpeechThreshold = 0.35,
    this.redemptionFrames = 8,
    this.preSpeechPadFrames = 1,
    this.endSpeechPadFrames = 1,
    this.frameSamples = 1536,
    this.minSpeechFrames = 3,
    this.numFramesToEmit = 0,
    this.submitUserSpeechOnPause = false,
    required this.onVadEvent,
    required this.model,
    this.baseAssetPath = 'https://cdn.jsdelivr.net/npm/@keyurmaru/vad@0.0.1/',
    this.onnxWASMBasePath =
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/',
    this.isDebug = false,
  });
}

class AudioNodeVAD {
  final web.AudioContext _context;
  final AudioNodeVadOptions _options;
  late final VadIteratorWeb _vadIterator;
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
    AudioNodeVadOptions options,
  ) async {
    final instance = AudioNodeVAD._(context, options);
    await instance._initialize();
    return instance;
  }

  Future<void> _initialize() async {
    if (_options.isDebug) {
      print('AudioNodeVAD._initialize: Starting initialization');
    }

    // Create VAD iterator
    _vadIterator = await VadIteratorWeb.create(
      isDebug: _options.isDebug,
      sampleRate: 16000,
      frameSamples: _options.frameSamples,
      positiveSpeechThreshold: _options.positiveSpeechThreshold,
      negativeSpeechThreshold: _options.negativeSpeechThreshold,
      redemptionFrames: _options.redemptionFrames,
      preSpeechPadFrames: _options.preSpeechPadFrames,
      minSpeechFrames: _options.minSpeechFrames,
      model: _options.model,
      baseAssetPath: _options.baseAssetPath,
      onnxWASMBasePath: _options.onnxWASMBasePath,
      endSpeechPadFrames: _options.endSpeechPadFrames,
      numFramesToEmit: _options.numFramesToEmit,
    );

    if (_options.isDebug) {
      print('AudioNodeVAD._initialize: VAD iterator created');
    }

    // Set the callback
    _vadIterator.setVadEventCallback(_options.onVadEvent);

    // Create resampler for audio processing
    if (_options.isDebug) {
      print(
          'AudioNodeVAD: Creating resampler with nativeSampleRate: ${_context.sampleRate.toInt()}, targetSampleRate: 16000');
    }
    _resampler = Resampler(ResamplerOptions(
      nativeSampleRate: _context.sampleRate.toInt(),
      targetSampleRate: 16000, // VAD models expect 16kHz
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
      _options.onVadEvent(VadEvent(
        type: VadEventType.error,
        timestamp: DateTime.now().millisecondsSinceEpoch / 1000,
        message: 'Failed to setup ScriptProcessorNode: $e',
      ));
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

          // Process through resampler
          final frames = _resampler.process(frame);
          for (final resampledFrame in frames) {
            _vadIterator.processFrame(resampledFrame);
          }

          if (_options.isDebug && _frameCount % 100 == 0) {
            print(
                'AudioNodeVAD: ScriptProcessor processed $_frameCount frames');
          }
        }
      } catch (e) {
        print('AudioNodeVAD: Error in ScriptProcessor audio processing: $e');
        _options.onVadEvent(VadEvent(
          type: VadEventType.error,
          timestamp: DateTime.now().millisecondsSinceEpoch / 1000,
          message: 'ScriptProcessor audio processing error: $e',
        ));
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
      _options.onVadEvent(VadEvent(
        type: VadEventType.error,
        timestamp: DateTime.now().millisecondsSinceEpoch / 1000,
        message:
            'AudioWorkletProcessor error: The processor encountered an error and will output silence. Please restart VAD.',
      ));

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

  void _handleAudioFrame(Map<Object?, Object?> data) async {
    if (!_active) return;

    try {
      final frameData = (data['frame'] as List).cast<double>();
      final frame = Float32List.fromList(frameData);

      _processedFrames++;

      // Process through resampler
      final frames = _resampler.process(frame);
      for (final resampledFrame in frames) {
        await _vadIterator.processFrame(resampledFrame);
      }
    } catch (e) {
      print('AudioNodeVAD: Error processing audio frame: $e');
      _options.onVadEvent(VadEvent(
        type: VadEventType.error,
        timestamp: DateTime.now().millisecondsSinceEpoch / 1000,
        message: 'Audio frame processing error: $e',
      ));
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

    if (_options.submitUserSpeechOnPause) {
      _vadIterator.forceEndSpeech();
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
    _vadIterator.reset();

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
    if (_options.submitUserSpeechOnPause) {
      _vadIterator.forceEndSpeech();
    }

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

    _vadIterator.release();
  }
}

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

  static Future<MicVAD> create(AudioNodeVadOptions options) async {
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

  void pause() {
    _audioNodeVAD.pause();
    _listening = false;
  }

  void start() {
    _audioNodeVAD.start();
    _listening = true;
  }

  void destroy() {
    if (_listening) {
      pause();
    }

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
}
