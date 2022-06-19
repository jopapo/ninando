import 'package:flutter/services.dart';
import 'package:collection/collection.dart';
import 'dart:math';
import 'dart:developer' as developer;

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

  double rootMeanSquare({int frameLength = 2048, int hopLength = 512}) {
    if (fixedData.length < frameLength) {
      throw ArgumentError.value(
          frameLength, "frameLength", "Frame can't be shorter than data size!");
    }
    if (hopLength < 1) {
      throw ArgumentError.value(
          hopLength, "hopLength", "Hop must be bigger than zero!");
    }

    int padding = frameLength ~/ 2; // truncating division

    // get the value from the padded list without changing the list (mode constant: border is zero)
    double _getPaddedCenterModeConstantByIndex(int index) {
      if (index >= padding) {
        index -= padding;
        if (index < fixedData.length) {
          return fixedData[index];
        }
      }
      return 0.0;
    }

    // framing loop
    int powCols = (fixedData.length ~/ hopLength) + 1;
    var sumPowColValues = List<double>.filled(powCols, 0);
    for (int frameIndex = 0; frameIndex < frameLength; frameIndex++) {
      for (int powIndex = 0; powIndex < powCols; powIndex++) {
        int currentIndex = (powIndex * hopLength) + frameIndex;
        double currentValue = _getPaddedCenterModeConstantByIndex(currentIndex);
        sumPowColValues[powIndex] += currentValue * currentValue; // pow 2
      }
    }

    return sumPowColValues.map<double>((e) => sqrt(e / frameLength)).average;
  }

  double zeroCrossingRate({int frameLength = 2048, int hopLength = 512}) {
    if (fixedData.length < frameLength) {
      throw ArgumentError.value(
          frameLength, "frameLength", "Frame can't be shorter than data size!");
    }
    if (hopLength < 1) {
      throw ArgumentError.value(
          hopLength, "hopLength", "Hop must be bigger than zero!");
    }

    int padding = frameLength ~/ 2; // truncating division

    // get the value from the padded list without changing the list (mode edge: border repeat first and last values)
    double _getPaddedCenterModeEdgeByIndex(int index) {
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

    // framing loop
    int zcrCols = (fixedData.length ~/ hopLength) + 1;
    var zeroCrossing = List<int>.filled(zcrCols, 0);
    for (int frameIndex = 1; frameIndex < frameLength; frameIndex++) {
      for (int zcrIndex = 0; zcrIndex < zcrCols; zcrIndex++) {
        int currentIndex = (zcrIndex * hopLength) + frameIndex;
        double currentValue = _getPaddedCenterModeEdgeByIndex(currentIndex);
        double previousValue =
            _getPaddedCenterModeEdgeByIndex(currentIndex - 1);
        if (currentValue.isNegative != previousValue.isNegative) {
          zeroCrossing[zcrIndex]++;
        }
      }
    }

    // zcr average
    return zeroCrossing.map<double>((e) => e / frameLength).average;
  }
}
