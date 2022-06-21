import 'dart:typed_data';

import 'package:fftea/fftea.dart';
import 'package:fftea/util.dart';
import 'package:flutter/services.dart';
import 'package:collection/collection.dart';
import 'dart:math';
import 'dart:developer' as developer;
import 'package:fftea/stft.dart';

class AudioEngineering {
  late List<double> _fixedData;
  late int _frameLength;
  late int _hopLength;

  AudioEngineering(Iterable<double> data,
      {int frameLength = 2048, int hopLength = 512}) {
    _fixedData = data is List<double> ? data : data.toList();

    if (_fixedData.length < frameLength) {
      throw ArgumentError.value(
          frameLength, "frameLength", "Frame can't be shorter than data size!");
    }
    if (hopLength < 1) {
      throw ArgumentError.value(
          hopLength, "hopLength", "Hop must be bigger than zero!");
    }

    _frameLength = frameLength;
    _hopLength = hopLength;
  }

  static Future<Iterable<double>> getTestData() async {
    return (await rootBundle
            .loadString('assets/ae_data/audio_engineering_audio_data.txt'))
        .split(",")
        .map<double>((e) => double.parse(e))
        .toList();
  }

  double rootMeanSquare() {
    var yFrame = PaddedFramedList(_fixedData, _frameLength, _hopLength);

    // framing loop
    var sumPowColValues = List<double>.filled(yFrame.cols, 0);
    yFrame.traverse().forEach((FrameData element) {
      sumPowColValues[element.colIndex] +=
          element.value * element.value; // pow 2
    });

    return sumPowColValues.map<double>((e) => sqrt(e / _frameLength)).average;
  }

  double zeroCrossingRate() {
    var yFrame = PaddedFramedList(_fixedData, _frameLength, _hopLength,
        padMode: PadMode.edge);

    var zeroCrossing = List<int>.filled(yFrame.cols, 0);
    yFrame.traverse(offsetFrameIndex: 1).forEach((FrameData element) {
      double currentValue = element.value;
      double previousValue = yFrame[element.index - 1];
      if (currentValue.isNegative != previousValue.isNegative) {
        zeroCrossing[element.colIndex]++;
      }
    });

    // zcr average
    return zeroCrossing.map<double>((e) => e / _frameLength).average;
  }

  double spectralCentroid() {
    final stft = STFT(_frameLength, Window.hanning(_frameLength));
    var yFrame = PaddedFramedList(_fixedData, _frameLength, _hopLength);

    //final spectrogram = <Float64List>[];
    //var list = yFrame.traverse().map<double>((e) => e.value).toList();
    var list = yFrame.toList();
    developer.log('list: ' + list.toString());
    stft.run(list, (Float64x2List freq) {
      //spectrogram.add(freq.magnitudes()); //.discardConjugates().magnitudes());
      developer.log('spectrogram: ' + freq.magnitudes().toString());
    }, _hopLength);

    final fft = FFT(_frameLength);

    // For some focking motive, this have only the same result when +1 (problem between librosa and scipy)
    var window = Window.hanning(_frameLength + 1).toList();
    developer.log('hann win: ' + window.toString());
    int windowIndex = 0;
    var stftMatrix = yFrame.traverse().map<double>((element) {
      var value = element.value * window[windowIndex].toDouble();
      windowIndex++;
      if (windowIndex >= window.length - 1) windowIndex = 0;
      return value;
    });

    developer.log('stftMatrix: ' + stftMatrix.toString());

    // var iterator = yFrame.traverse().iterator;
    // while (iterator.moveNext()) {
    //   var chunk =
    //       Float64x2List.fromList(List<double>.generate(_frameLength, (index) {
    //     var current = iterator.current;
    //     iterator.moveNext();
    //     return current.value;
    //   }).map<Float64x2>((e) => Float64x2(e, 0)).toList());

    //   //window.inPlaceApplyWindow(chunk);
    //   //fft.inPlaceFft(chunk);

    //   developer.log('spectrogram - my chunks: ' + chunk.toString());
    // }

    //developer.log('spectrogram: ' + spectrogram.toString());

    return 0.0;
  }
}

class PaddedFramedList extends Iterable<double> {
  late int _frameLength;
  late int _hopLength;
  late int _padding;
  late List<double> _baseList;
  late int _cols;
  late int _length;
  double _startPadFill = 0.0;
  double _endPadFill = 0.0;

  int get length => _length;
  int get frameLength => _frameLength;
  int get hopLength => _hopLength;
  int get cols => _cols;

  PaddedFramedList(List<double> baseList, int frameLength, int hopLength,
      {PadMode padMode = PadMode.zero}) {
    _baseList = baseList;
    _frameLength = frameLength;
    _hopLength = hopLength;
    _padding = frameLength ~/ 2;
    if (padMode == PadMode.edge) {
      _startPadFill = _baseList.first;
      _endPadFill = _baseList.last;
    }

    _cols = (_baseList.length ~/ _hopLength) + 1;
    _length = _frameLength * _cols;
  }

  Iterable<FrameData> traverse({int offsetFrameIndex = 0}) sync* {
    for (int frameIndex = offsetFrameIndex;
        frameIndex < _frameLength;
        frameIndex++) {
      for (int colIndex = 0; colIndex < _cols; colIndex++) {
        int currentIndex = (colIndex * _hopLength) + frameIndex;
        double currentValue = this[currentIndex];
        yield FrameData(currentIndex, currentValue, colIndex, frameIndex);
      }
    }
  }

  double operator [](int index) {
    if (index < _padding) {
      return _startPadFill;
    } else {
      index -= _padding;
      if (index < _baseList.length) {
        return _baseList[index];
      } else {
        return _endPadFill;
      }
    }
  }

  @override
  Iterator<double> get iterator => List<double>.filled(_padding, _startPadFill)
      .followedBy(_baseList)
      .followedBy(List<double>.filled(_padding, _endPadFill))
      .iterator;
}

class FrameData {
  int index;
  double value;
  int colIndex;
  int frameIndex;

  FrameData(this.index, this.value, this.colIndex, this.frameIndex);
}

enum PadMode { edge, zero }
