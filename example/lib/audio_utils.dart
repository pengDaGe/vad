// lib/audio_utils.dart
// audio_utils.dart

// Dart imports:
import 'dart:convert';
import 'dart:typed_data';

class AudioUtils {
  static String createWavUrl(List<double> samples) {
    final wavData = float32ToWav(samples);
    final base64Wav = base64Encode(wavData);
    return 'data:audio/wav;base64,$base64Wav';
  }

  static Uint8List float32ToWav(List<double> float32Array) {
    const int sampleRate = 16000;
    const int byteRate = sampleRate * 2; // 16-bit = 2 bytes per sample
    final int totalAudioLen = float32Array.length * 2;
    final int totalDataLen = totalAudioLen + 36;

    final ByteData buffer = ByteData(44 + totalAudioLen);

    // Write WAV header
    _writeString(buffer, 0, 'RIFF');
    buffer.setInt32(4, totalDataLen, Endian.little);
    _writeString(buffer, 8, 'WAVE');
    _writeString(buffer, 12, 'fmt ');
    buffer.setInt32(16, 16, Endian.little);
    buffer.setInt16(20, 1, Endian.little);
    buffer.setInt16(22, 1, Endian.little);
    buffer.setInt32(24, sampleRate, Endian.little);
    buffer.setInt32(28, byteRate, Endian.little);
    buffer.setInt16(32, 2, Endian.little);
    buffer.setInt16(34, 16, Endian.little);
    _writeString(buffer, 36, 'data');
    buffer.setInt32(40, totalAudioLen, Endian.little);

    // Convert and write audio data
    int offset = 44;
    for (double sample in float32Array) {
      sample = sample.clamp(-1.0, 1.0);
      final int pcm = (sample < 0 ? sample * 0x8000 : sample * 0x7FFF).toInt();
      buffer.setInt16(offset, pcm, Endian.little);
      offset += 2;
    }

    return buffer.buffer.asUint8List();
  }

  static void _writeString(ByteData view, int offset, String string) {
    for (int i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.codeUnitAt(i));
    }
  }
}
