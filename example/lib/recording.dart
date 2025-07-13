// lib/recording.dart
enum RecordingType {
  speechStart,
  realSpeechStart,
  speechEnd,
  misfire,
  error,
}

class Recording {
  final List<double>? samples;
  final RecordingType type;
  final DateTime timestamp;

  Recording({
    this.samples,
    required this.type,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}
