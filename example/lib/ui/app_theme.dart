// lib/ui/app_theme.dart

// Flutter imports:
import 'package:flutter/material.dart';

class AppTheme {
  static ThemeData getDarkTheme() {
    return ThemeData.dark().copyWith(
      primaryColor: Colors.blue,
      scaffoldBackgroundColor: const Color(0xFF121212),
      cardColor: const Color(0xFF1E1E1E),
      sliderTheme: SliderThemeData(
        activeTrackColor: Colors.blue[400],
        inactiveTrackColor: Colors.grey[800],
        thumbColor: Colors.blue[300],
        overlayColor: Colors.blue.withAlpha(32),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.blue[700],
          foregroundColor: Colors.white,
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: Colors.blue[300],
        ),
      ),
    );
  }
}
