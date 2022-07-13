import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'dart:developer' as developer;

import 'package:complex/complex.dart';
import 'package:fftea/fftea.dart';
import 'package:flutter/services.dart';
import 'package:collection/collection.dart';
import 'package:fftea/stft.dart';
import 'package:path_provider/path_provider.dart';

class AudioEngineering {
  late List<double> _fixedData;
  late int _frameLength;
  late int _hopLength;
  final Map<int, Complex> _cachedSenoids = <int, Complex>{};
  late final List<double> _cachedFft = fft().toList();
  late final List<double> _cachedWindow = window().toList();

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
    var list = (await rootBundle
            .loadString('assets/ae_data/audio_engineering_audio_data.txt'))
        .split(",")
        .map<double>((e) => double.parse(e))
        .toList();

    developer.log('test data size = ${list.length}');

    return list;
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

  Iterable<double> window() {
    // This have only the same result than librosa when +1.
    // Librosa and scipy have the parameter sym, where is True by default - differente from here (issue explaining below)
    // Issue: https://github.com/librosa/librosa/issues/1510#event-6846800459
    var window = Window.hanning(_frameLength + 1);
    return Float64List.sublistView(window, 0, _frameLength)
        .map((e) => (e * 100.0).roundToDouble() / 100.0);
  }

  /// Reimplemantation of Librosa/Numpy Fast Fourier Transform: https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html#numpy.fft.fft
  Iterable<double> fft() {
    // No need for cache. Small footprint.
    var yFrame = PaddedFramedList(_fixedData, _frameLength, _hopLength);

    // Soma e multiplicação de números complexos
    //https://www.ufrgs.br/reamat/TransformadasIntegrais/livro-af/rdnceft-nx00fameros_complexos_e_fx00f3rmula_de_euler.html
    var N = _frameLength;
    var n = (N ~/ 2 + 1);
    var compList = List<Complex>.filled(n * N, Complex.zero, growable: false);

    for (int k = 0; k < n; k++) {
      yFrame.traverse().forEach((element) {
        var m = element.frameIndex;
        var pos = m * k;
        var c1 = _cachedSenoids[pos];
        if (c1 == null) {
          // e^(-2i*pi*m*k/N) -> euller -> e^(ix) = cos(x) + i*sin(x)
          var exponent = -2 * pi * pos / N;
          c1 = Complex(cos(exponent), sin(exponent));
          _cachedSenoids[pos] = c1;
        }
        var a2 = element.value * _cachedWindow[element.frameIndex];
        //var c2 = Complex(a2, 0);
        compList[k * _frameLength + element.colIndex] += (c1 * a2);
      });
    }

    return compList.map<double>((c) => c.abs());
  }

  Future<double> spectralCentroid() async {
    final Directory directory = await getApplicationDocumentsDirectory();
    final File file = File('${directory.path}/fft_test_data.txt');
    file.writeAsStringSync(_cachedFft.toString());

    developer.log("fft: len=${_cachedFft.length}; avg=${_cachedFft.average}");

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

  @override
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

/// https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
extension StandardScaler on List<double> {
  static final List<double> _featureMeans = [
    0.049971,
    0.110803,
    -311.205094,
    112.776219,
    -9.223425,
    18.414529,
    -4.045880,
    3.435483,
    0.790667,
    3.239740,
    1.152651,
    2.463913,
    -0.064407,
    -0.111859,
    0.114910,
    2412.593058,
    6.792487e+03,
    3.214534e+03
  ];

  static final List<double> _featureVars = [
    0.001558,
    0.007403,
    8031.033092,
    1859.978735,
    1885.003648,
    247.324044,
    294.303209,
    230.088700,
    199.573403,
    150.746311,
    138.797059,
    105.697523,
    88.285784,
    70.155498,
    65.911219,
    711495.260635,
    5.683198e+06,
    1.362893e+06
  ];

  List<double> standardScale() {
    var result = List<double>.generate(
        length,
        (index) =>
            (this[index] - _featureMeans[index]) / sqrt(_featureVars[index]));
    (length);
    //developer.log("standardScaler $result");
    return result;
  }
}
