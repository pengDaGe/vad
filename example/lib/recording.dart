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

  Recording({
    this.samples,
    required this.type,
    DateTime? timestamp,
    this.chunkIndex,
  }) : timestamp = timestamp ?? DateTime.now();
}
