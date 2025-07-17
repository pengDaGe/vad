import 'dart:ffi';
import 'dart:io';
import 'package:vad/src/platform/native/bindings/onnxruntime_bindings_generated.dart';

String _getArchitecture() {
  // Check if we're running on ARM64/AArch64
  final processorInfo = Platform.version.toLowerCase();
  if (processorInfo.contains('arm64') || processorInfo.contains('aarch64')) {
    return 'arm64';
  }
  return 'x64';
}

final DynamicLibrary _dylib = () {
  if (Platform.isAndroid) {
    return DynamicLibrary.open('libonnxruntime.so');
  }

  if (Platform.isIOS) {
    return DynamicLibrary.process();
  }

  if (Platform.isMacOS) {
    return DynamicLibrary.process();
  }

  if (Platform.isWindows) {
    final arch = _getArchitecture();
    try {
      return DynamicLibrary.open('$arch/onnxruntime.dll');
    } catch (e) {
      throw UnsupportedError('ONNX Runtime library not found for Windows $arch. Expected: $arch/onnxruntime.dll');
    }
  }

  if (Platform.isLinux) {
    final arch = _getArchitecture();
    try {
      return DynamicLibrary.open('$arch/libonnxruntime.so.1.22.0');
    } catch (e) {
      throw UnsupportedError('ONNX Runtime library not found for Linux $arch. Expected: $arch/libonnxruntime.so.1.22.0');
    }
  }

  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

/// ONNX Runtime Bindings for VAD
final onnxRuntimeBinding = OnnxRuntimeBindings(_dylib);
