import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_sound/public/tau.dart';
import 'package:flutter_sound/public/util/flutter_sound_helper.dart';

import 'dart:developer' as developer;

class BatchTransformer implements StreamTransformer<Food, List<double>> {
  final StreamController<List<double>> _controller =
      StreamController<List<double>>();
  List<double> data = List<double>.empty(growable: true);
  late int tSampleRate;

  BatchTransformer(sampleRate) {
    tSampleRate = sampleRate;
  }

  @override
  Stream<List<double>> bind(Stream<Food> stream) {
    int limits = tSampleRate * 5; // 5s

    stream.listen((buffer) {
      // Filha da puta! Vem com envelope. Demorei pra cacete pra descobrir.
      Uint8List pcmBuffer = flutterSoundHelper.waveToPCMBuffer(
          inputBuffer: (buffer as FoodData).data!);
      var pcm16 = pcmBuffer.buffer.asInt16List();
      // asInt16List();
      int dataTo16Size = data.length;
      int totalSize = dataTo16Size + pcm16.length;
      if (totalSize <= limits) {
        data.addAll(pcm16.normalize());
      } else {
        int fitSize = limits - dataTo16Size;
        data.addAll(pcm16.take(fitSize).normalize());

        developer.log('C: dataSize: ' + data.length.toString());

        _controller.add(data);
        data = List<double>.empty(growable: true);
        data.addAll(pcm16.skip(fitSize).normalize());
      }
    }).onDone(() {
      developer.log('B: dataSize: ' + data.length.toString());
      if (data.isNotEmpty) {
        _controller.add(data);
        data = List<double>.empty(growable: true);
      }
    });

    // return an output stream for our listener
    return _controller.stream;
  }

  @override
  StreamTransformer<RS, RT> cast<RS, RT>() {
    return StreamTransformer.castFrom(this);
  }
}

extension Normalizing16bits on Iterable<int> {
  // https://libsndfile.github.io/libsndfile/FAQ.html#Q010
  static double normFactor = 0x8000;

  double _normalizeInt16ToFloat(int n) {
    return n / normFactor;
  }

  Iterable<double> normalize() {
    return map<double>(_normalizeInt16ToFloat);
  }
}
