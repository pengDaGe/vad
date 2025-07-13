// lib/src/web/onnx_runtime_web.dart

// ignore_for_file: public_member_api_docs, avoid_print

// Dart imports:
import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

// Project imports:
import '../model_utils.dart';
import '../vad_event.dart';

@JS('ort')
external OrtGlobal get ort;

@JS()
extension type OrtGlobal._(JSObject _) implements JSObject {
  external EnvNamespace get env;
  // ignore: non_constant_identifier_names
  external InferenceSessionConstructor get InferenceSession;
  // ignore: non_constant_identifier_names
  external TensorConstructor get Tensor;
}

@JS()
extension type EnvNamespace._(JSObject _) implements JSObject {
  external WasmNamespace get wasm;
}

@JS()
extension type WasmNamespace._(JSObject _) implements JSObject {
  external set wasmPaths(String paths);
}

@JS()
extension type InferenceSessionConstructor._(JSObject _) implements JSObject {
  external JSPromise<InferenceSession> create(JSAny modelData);
}

@JS()
extension type TensorConstructor._(JSObject _) implements JSObject {
  external Tensor create(
      String type, JSFloat32Array data, JSArray<JSNumber> dims);
}

@JS()
extension type InferenceSession._(JSObject _) implements JSObject {
  external JSPromise<RunResult> run(JSObject feeds);
}

@JS()
extension type Tensor._(JSObject _) implements JSObject {
  external JSFloat32Array get data;
  external JSArray<JSNumber> get dims;
}

@JS()
extension type RunResult._(JSObject _) implements JSObject {
  external Tensor operator [](String key);
}

abstract class VadModel {
  Future<SpeechProbabilities> process(Float32List frame);
  void resetState();
}

class SileroV5Model implements VadModel {
  final InferenceSession _session;
  final Map<String, String> _inputNames;
  final Map<String, String> _outputNames;
  Tensor? _state;
  Tensor? _sr;
  Future<void>? _currentProcessing;
  int _processCount = 0;

  SileroV5Model._(this._session, this._inputNames, this._outputNames) {
    resetState();
  }

  static Future<SileroV5Model> create(String modelUrl,
      [String? onnxWASMBasePath]) async {
    try {
      final wasmPath = onnxWASMBasePath ??
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
      final normalizedWasmPath =
          wasmPath.endsWith('/') ? wasmPath : '$wasmPath/';
      ort.env.wasm.wasmPaths = normalizedWasmPath;
      final session = await ort.InferenceSession.create(modelUrl.toJS).toDart;
      final inputNames = getModelInputNames('v5');
      final outputNames = getModelOutputNames('v5');

      return SileroV5Model._(session, inputNames, outputNames);
    } catch (e) {
      print('Error creating SileroV5Model: $e');
      rethrow;
    }
  }

  @override
  void resetState() {
    final zeroes = Float32List(2 * 1 * 128);
    _state = (ort.Tensor as JSFunction).callAsConstructor(
      'float32'.toJS,
      zeroes.toJS,
      [2, 1, 128].map((e) => e.toJS).toList().toJS,
    ) as Tensor;

    final srArray = [16000].map((e) => e.toJS).toList().toJS;
    _sr = (ort.Tensor as JSFunction).callAsConstructor(
      'int64'.toJS,
      srArray,
      [1].map((e) => e.toJS).toList().toJS,
    ) as Tensor;
  }

  @override
  Future<SpeechProbabilities> process(Float32List frame) async {
    await _currentProcessing;
    final processingFuture = _processFrameInternal(frame);
    _currentProcessing = processingFuture.then((_) {});
    return processingFuture;
  }

  Future<SpeechProbabilities> _processFrameInternal(Float32List frame) async {
    try {
      _processCount++;

      final inputTensor = (ort.Tensor as JSFunction).callAsConstructor(
        'float32'.toJS,
        frame.toJS,
        [1, frame.length].map((e) => e.toJS).toList().toJS,
      ) as Tensor;

      final feeds = {
        _inputNames['input']!: inputTensor,
        _inputNames['state']!: _state!,
        _inputNames['sr']!: _sr!,
      }.jsify()! as JSObject;

      final results = await _session.run(feeds).toDart;

      final output = results[_outputNames['output']!];
      final newState = results[_outputNames['state']!];

      _state = newState;

      final prob = (output.data.toDart)[0];
      return SpeechProbabilities(
        isSpeech: prob,
        notSpeech: 1.0 - prob,
      );
    } catch (e) {
      print('Error in SileroV5Model.process: $e');
      print('Process count when error occurred: $_processCount');
      rethrow;
    }
  }
}

class SileroV4Model implements VadModel {
  final InferenceSession _session;
  final Map<String, String> _inputNames;
  final Map<String, String> _outputNames;
  Tensor? _h;
  Tensor? _c;
  Tensor? _sr;
  Future<void>? _currentProcessing;
  int _processCount = 0;

  SileroV4Model._(this._session, this._inputNames, this._outputNames) {
    resetState();
  }

  static Future<SileroV4Model> create(String modelUrl,
      [String? onnxWASMBasePath]) async {
    try {
      final wasmPath = onnxWASMBasePath ??
          'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
      final normalizedWasmPath =
          wasmPath.endsWith('/') ? wasmPath : '$wasmPath/';

      ort.env.wasm.wasmPaths = normalizedWasmPath;

      final session = await ort.InferenceSession.create(modelUrl.toJS).toDart;
      final inputNames = getModelInputNames('legacy');
      final outputNames = getModelOutputNames('legacy');

      return SileroV4Model._(session, inputNames, outputNames);
    } catch (e) {
      print('Error creating SileroV4Model: $e');
      rethrow;
    }
  }

  @override
  void resetState() {
    final zeroes = Float32List(2 * 1 * 64);
    _h = (ort.Tensor as JSFunction).callAsConstructor(
      'float32'.toJS,
      zeroes.toJS,
      [2, 1, 64].map((e) => e.toJS).toList().toJS,
    ) as Tensor;

    _c = (ort.Tensor as JSFunction).callAsConstructor(
      'float32'.toJS,
      zeroes.toJS,
      [2, 1, 64].map((e) => e.toJS).toList().toJS,
    ) as Tensor;

    final srArray = [16000].map((e) => e.toJS).toList().toJS;
    _sr = (ort.Tensor as JSFunction).callAsConstructor(
      'int64'.toJS,
      srArray,
      [1].map((e) => e.toJS).toList().toJS,
    ) as Tensor;
  }

  @override
  Future<SpeechProbabilities> process(Float32List frame) async {
    await _currentProcessing;
    final processingFuture = _processFrameInternal(frame);
    _currentProcessing = processingFuture.then((_) {});
    return processingFuture;
  }

  Future<SpeechProbabilities> _processFrameInternal(Float32List frame) async {
    try {
      _processCount++;

      final inputTensor = (ort.Tensor as JSFunction).callAsConstructor(
        'float32'.toJS,
        frame.toJS,
        [1, frame.length].map((e) => e.toJS).toList().toJS,
      ) as Tensor;

      final feeds = {
        _inputNames['input']!: inputTensor,
        _inputNames['h']!: _h!,
        _inputNames['c']!: _c!,
        _inputNames['sr']!: _sr!,
      }.jsify()! as JSObject;

      final results = await _session.run(feeds).toDart;

      final output = results[_outputNames['output']!];
      final newH = results[_outputNames['h']!];
      final newC = results[_outputNames['c']!];

      _h = newH;
      _c = newC;

      final prob = (output.data.toDart)[0];
      return SpeechProbabilities(
        isSpeech: prob,
        notSpeech: 1.0 - prob,
      );
    } catch (e) {
      print('Error in SileroV4Model.process: $e');
      print('Process count when error occurred: $_processCount');
      rethrow;
    }
  }
}
