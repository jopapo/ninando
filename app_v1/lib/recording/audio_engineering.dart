import 'package:flutter/services.dart';
import 'package:collection/collection.dart';

class AudioEngineering {
  late List<double> fixedData;

  AudioEngineering(Iterable<double> data) {
    fixedData = data is List<double> ? data : data.toList();
  }

  static Future<Iterable<double>> getTestData() async {
    return (await rootBundle
            .loadString('assets/ae_data/audio_engineering_audio_data.txt'))
        .split(",")
        .map<double>((e) => double.parse(e))
        .toList();
  }

  double zcr({int frameLength = 2048, int hopLength = 512}) {
    int padding = frameLength ~/ 2; // truncating division

    double getByIndex(int index) {
      if (index < padding) {
        return fixedData.first;
      } else {
        index -= padding;
        if (index < fixedData.length) {
          return fixedData[index];
        } else {
          return fixedData.last;
        }
      }
    }

    int zcrCols = (fixedData.length ~/ hopLength) + 1;
    var zeroCrossing = List<int>.filled(zcrCols, 0);
    for (int frameIndex = 1; frameIndex < frameLength; frameIndex++) {
      for (int zcrIndex = 0; zcrIndex < zcrCols; zcrIndex++) {
        int currentIndex = (zcrIndex * hopLength) + frameIndex;
        double currentValue = getByIndex(currentIndex);
        double previousValue = getByIndex(currentIndex - 1);
        if (currentValue.isNegative != previousValue.isNegative) {
          zeroCrossing[zcrIndex]++;
        }
      }
    }

    return zeroCrossing.map<double>((e) => e / frameLength).average;
  }
}
