// lib/recording.dart
enum RecordingType {
  speechStart,
  realSpeechStart,
  speechEnd,
  misfire,
  error,
  chunk,
}

class Recording {
  final List<double>? samples;
  final RecordingType type;
  final DateTime timestamp;
  final int? chunkIndex; // For tracking chunk sequence
  final bool? isFinal; // For tracking final chunk in sequence

  Recording({
    this.samples,
    required this.type,
    DateTime? timestamp,
    this.chunkIndex,
    this.isFinal,
  }) : timestamp = timestamp ?? DateTime.now();
}
