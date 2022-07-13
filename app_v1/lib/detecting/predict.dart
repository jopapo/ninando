import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:chaquopy/chaquopy.dart';
import 'package:flutter/material.dart';
import 'package:flutter_sound/public/tau.dart';
import 'package:ninando/recording/audio_engineering.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:developer' as developer;

// ignore: import_of_legacy_library_into_null_safe
import 'package:sklite/SVM/SVM.dart';
// ignore: import_of_legacy_library_into_null_safe
import 'package:sklite/utils/io.dart';

/// Widget for prediction
class PredictionWidget extends StatefulWidget {
  late final Future<SVC> _customModel = loadSvcModel();

  PredictionWidget({Key? key}) : super(key: key);

  Future<SVC> loadSvcModel() async {
    var assetPath = "assets/models/svc-1.json";
    var modelData = await loadModel(assetPath);
    var svc = SVC.fromMap(json.decode(modelData));
    return svc;
  }

  final StreamController<int> onNewPrediction = StreamController<int>();
  final StreamController<File> onRecordingStopped = StreamController<File>();
  final StreamController<bool> onAlertThreashold = StreamController<bool>();

  Future<File> getNewOutputFile() async {
    var tempDir = await getTemporaryDirectory();
    var suffix =
        DateTime.now().toIso8601String().replaceAll(RegExp(r'[:\.]'), '_');
    var sinkFile = '${tempDir.path}/flutter_sound_realtime_$suffix.pcm';

    var outputFile = File(sinkFile);

    return outputFile;
  }

  Future<StreamSubscription<Food>> onNewRecordingSubscription(
      StreamController<Food> foodController, int sampleRate) async {
    var outputFile = await getNewOutputFile();
    var sink = outputFile.openWrite(mode: FileMode.writeOnly);

    File? transitionFile;
    IOSink? transitionSink;
    var transitionTotalSize = 0.0;
    //var transitions = 0;
    const transitionThreashold = 60; // 1 minute

    var totalSize = 0.0;
    var lastStartSecond = -1;
    var fullSampleRate = sampleRate * 2; // for the seconds calculation
    const sampleSeconds = 5;

    var subscription = foodController.stream.listen(
      (buffer) {
        sink.add((buffer as FoodData).data!);
        transitionSink?.add(buffer.data!);
        totalSize += buffer.data!.length;
        transitionTotalSize += buffer.data!.length;
        var newStartSecond = (totalSize ~/ fullSampleRate) -
            sampleSeconds; // Need to be always 5 seconds behind
        if (newStartSecond > lastStartSecond) {
          lastStartSecond = newStartSecond;

          predict(outputFile, newStartSecond)
              .then((prediction) => onNewPrediction.add(prediction));

          // Começa um novo arquivo a cada 1 minuto pra evitar estouro de disco
          if (newStartSecond >= transitionThreashold) {
            if (newStartSecond == transitionThreashold) {
              getNewOutputFile().then((newFile) {
                transitionTotalSize = 0;
                transitionFile = newFile;
                transitionSink =
                    transitionFile!.openWrite(mode: FileMode.writeOnly);
              });
            }
            if (newStartSecond > transitionThreashold + sampleSeconds) {
              sink.close().then((_) {
                //outputFile.rename(outputFile.path + '_part-${++transitions}');
                outputFile.delete();

                developer.log(
                    "transitioningFile: ${outputFile.path} -> ${transitionFile!.path}");

                sink = transitionSink!;
                outputFile = transitionFile!;
                transitionSink = null;
                transitionFile = null;
                totalSize = transitionTotalSize;
                lastStartSecond = -1;
              });
            }
          }
        }
      },
    );

    foodController.onCancel = () {
      sink.close().then((_) {
        //outputFile.rename(outputFile.path + '_stopped');
        outputFile.delete();
      });
      onRecordingStopped.add(outputFile);
    };

    return subscription;
  }

  Future<int> predict(File file, int tick) {
    return Chaquopy.executeCode("${file.path}|$tick").then((result) {
      //developer.log("analysisDetected: ${file.path}|$tick; result: $result");

      var textResult = result['textOutputOrError'].toString();
      if (textResult.startsWith("Error:")) {
        throw Exception(textResult.substring(6));
      }

      var data = textResult.trim().split(' ');

      if (data.length == 18) {
        var means = List<double>.generate(
            data.length, (index) => double.parse(data[index]));

        return _customModel.then((svc) {
          return svc.predict(means.standardScale());
        });
      }

      return -1;
    });
  }

  @override
  _PredictionWidgetState createState() => _PredictionWidgetState();
}

class _PredictionWidgetState extends State<PredictionWidget> {
  final List<int> _detectionRange = List<int>.empty(growable: true);
  int alertLevel = 0;
  int silenceCount = 0;
  final int alertThreashold = 3;

  @override
  void initState() {
    super.initState();

    widget.onNewPrediction.stream.listen((prediction) {
      _detectionRange.add(prediction);

      var wasAlerted = alertLevel >= alertThreashold;
      if (prediction > 0) {
        alertLevel++;
        silenceCount = 0;
      } else {
        silenceCount++;
        if (silenceCount > 10) alertLevel = 0;
      }

      var isAlerted = alertLevel >= alertThreashold;
      if (wasAlerted != isAlerted) {
        widget.onAlertThreashold.add(isAlerted);
      }

      setState(() {
        while (_detectionRange.length > 10) {
          _detectionRange.removeAt(0);
        }
      });
    });

    widget.onRecordingStopped.stream.listen((event) {
      setState(() {
        _detectionRange.clear();
      });
    });
  }

  @override
  void dispose() {
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _body(),
    );
  }

  Widget _body() {
    return Container(
      margin: const EdgeInsets.all(3),
      padding: const EdgeInsets.all(3),
      height: 80,
      width: double.infinity,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        border: Border.all(
          color: Colors.indigo,
          width: 3,
        ),
      ),
      child: Text("$_detectionRange - #$alertLevel",
          style: alertLevel >= alertThreashold
              ? const TextStyle(color: Colors.red)
              : null),
    );
  }
}
