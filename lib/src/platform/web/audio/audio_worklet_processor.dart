// lib/src/platform/web/audio/audio_worklet_processor.dart
//
// This file contains the AudioWorklet processor code generation for web platform.
// The processor handles real-time audio processing in a separate thread.

// ignore_for_file: public_member_api_docs

/// Generates the JavaScript code for the AudioWorklet processor.
/// This processor runs in a separate audio thread and handles real-time
/// audio buffering and frame extraction.
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