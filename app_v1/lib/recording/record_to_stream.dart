/*
 * Copyright 2018, 2019, 2020, 2021 Dooboolab.
 *
 * This file is part of Flutter-Sound.
 *
 * Flutter-Sound is free software: you can redistribute it and/or modify
 * it under the terms of the Mozilla Public License version 2 (MPL2.0),
 * as published by the Mozilla organization.
 *
 * Flutter-Sound is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * MPL General Public License for more details.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:ninando/recording/audio_engineering.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:developer' as developer;
import 'batch_transformer.dart';

/*
 * This is an example showing how to record to a Dart Stream.
 * It writes all the recorded data from a Stream to a File, which is completely stupid:
 * if an App wants to record something to a File, it must not use Streams.
 *
 * The real interest of recording to a Stream is for example to feed a
 * Speech-to-Text engine, or for processing the Live data in Dart in real time.
 *
 */

///
const int tSampleRate = 44100;
const int tBitRate = 16000;
typedef _Fn = void Function();

/// Example app.
class RecordToStreamExample extends StatefulWidget {
  @override
  _RecordToStreamExampleState createState() => _RecordToStreamExampleState();
}

class _RecordToStreamExampleState extends State<RecordToStreamExample> {
  FlutterSoundPlayer? _mPlayer = FlutterSoundPlayer();
  FlutterSoundRecorder? _mRecorder = FlutterSoundRecorder();
  bool _mPlayerIsInited = false;
  bool _mRecorderIsInited = false;
  bool _mplaybackReady = false;
  String? _mPath;
  StreamSubscription? _mRecordingDataSubscription;
  StreamSubscription<RecordingDisposition>? _mRecordingProgressSubscription;

  int pos = 0;
  double dbLevel = 0;
  final spectrogram = <Float64List>[];

  Future<void> _openRecorder() async {
    // Zera caso reinicie
    pos = 0;
    dbLevel = 0;

    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw RecordingPermissionException('Microphone permission not granted');
    }

    await _mRecorder!.openAudioSession();

    // final session = await AudioSession.instance;
    // await session.configure(AudioSessionConfiguration(
    //   avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
    //   avAudioSessionCategoryOptions:
    //       AVAudioSessionCategoryOptions.allowBluetooth |
    //           AVAudioSessionCategoryOptions.defaultToSpeaker,
    //   avAudioSessionMode: AVAudioSessionMode.spokenAudio,
    //   avAudioSessionRouteSharingPolicy:
    //       AVAudioSessionRouteSharingPolicy.defaultPolicy,
    //   avAudioSessionSetActiveOptions: AVAudioSessionSetActiveOptions.none,
    //   androidAudioAttributes: const AndroidAudioAttributes(
    //     contentType: AndroidAudioContentType.speech,
    //     flags: AndroidAudioFlags.none,
    //     usage: AndroidAudioUsage.voiceCommunication,
    //   ),
    //   androidAudioFocusGainType: AndroidAudioFocusGainType.gain,
    //   androidWillPauseWhenDucked: true,
    // ));

    setState(() {
      _mRecorderIsInited = true;
    });

    _mRecorder!.setSubscriptionDuration(const Duration(milliseconds: 250));
    _mRecordingProgressSubscription = _mRecorder!.onProgress!.listen((e) {
      setState(() {
        pos = e.duration.inMilliseconds;
        if (e.decibels != null) {
          dbLevel = e.decibels as double;
        }
      });
    });
  }

  @override
  void initState() {
    // Timer.periodic(const Duration(seconds: 1), (timer) {
    //   setState(
    //       () {}); // Só pra forçar atualização, pois não quero pra cada fluxo de dados
    // });

    super.initState();
    // Be careful : openAudioSession return a Future.
    // Do not access your FlutterSoundPlayer or FlutterSoundRecorder before the completion of the Future
    _mPlayer!.openAudioSession().then((value) {
      setState(() {
        _mPlayerIsInited = true;
      });
    });
    _openRecorder(); //.then((value) => _registerEvents());
  }

  @override
  void dispose() {
    stopPlayer();
    _mPlayer!.closeAudioSession();
    _mPlayer = null;

    stopRecorder();
    _mRecorder!.closeAudioSession();
    _mRecorder = null;

    _mRecordingDataSubscription!.cancel();
    _mRecordingDataSubscription = null;
    _mRecordingProgressSubscription!.cancel();
    _mRecordingProgressSubscription = null;

    super.dispose();
  }

  Future<IOSink> createSink() async {
    //var tempDir = await getTemporaryDirectory();
    var tempDir = await getApplicationDocumentsDirectory();
    _mPath = '${tempDir.path}/flutter_sound_example.pcm';
    var outputFile = File(_mPath!);
    if (outputFile.existsSync()) {
      await outputFile.delete();
    }
    return outputFile.openWrite();
  }

  // ----------------------  Here is the code to record to a Stream ------------

  Future<void> record() async {
    assert(_mRecorderIsInited && _mPlayer!.isStopped);

    // var sink = await createSink();
    var recordingDataController = StreamController<Food>();
    var batchTransformer = BatchTransformer(tSampleRate);

    final Directory directory = await getApplicationDocumentsDirectory();
    final File file = File('${directory.path}/float_data_from_device.txt');

    AudioEngineering test =
        //AudioEngineering(await AudioEngineering.getTestData());
        AudioEngineering(<double>[-1, -2, -3, -4, 0, 6, 7, -8, 9, 9],
            frameLength: 6, hopLength: 2);

    _mRecordingDataSubscription = recordingDataController.stream
        .transform(batchTransformer)
        .listen((event) {
      file.writeAsStringSync(event.toString());

      developer.log("batch: " +
          event.length.toString() +
          ", take 100: " +
          event.take(100).toString());

      //developer.log("zcr:" + test.zeroCrossingRate().toString());
      //developer.log("rms:" + test.rootMeanSquare().toString());
      developer.log("spectral_centroid:" + test.spectralCentroid().toString());
    });

    await _mRecorder!.startRecorder(
        toStream: recordingDataController.sink,
        codec: Codec.pcm16,
        numChannels: 1, // Mono
        sampleRate: tSampleRate,
        bitRate: tBitRate);

    setState(() {});
  }
  // --------------------- (it was very simple, wasn't it ?) -------------------

  Future<void> stopRecorder() async {
    await _mRecorder!.stopRecorder();
    if (_mRecordingDataSubscription != null) {
      await _mRecordingDataSubscription!.cancel();
      _mRecordingDataSubscription = null;
    }
    _mplaybackReady = true;
  }

  _Fn? getRecorderFn() {
    if (!_mRecorderIsInited || !_mPlayer!.isStopped) {
      return null;
    }
    return _mRecorder!.isStopped
        ? record
        : () {
            stopRecorder().then((value) => setState(() {}));
          };
  }

  void play() async {
    assert(_mPlayerIsInited &&
        _mplaybackReady &&
        _mRecorder!.isStopped &&
        _mPlayer!.isStopped);
    await _mPlayer!.startPlayer(
        fromURI: _mPath,
        sampleRate: tSampleRate,
        codec: Codec.pcm16,
        numChannels: 1, // Mono
        whenFinished: () {
          setState(() {});
        }); // The readability of Dart is very special :-(
    setState(() {});
  }

  Future<void> stopPlayer() async {
    await _mPlayer!.stopPlayer();
  }

  _Fn? getPlaybackFn() {
    if (!_mPlayerIsInited || !_mplaybackReady || !_mRecorder!.isStopped) {
      return null;
    }
    return _mPlayer!.isStopped
        ? play
        : () {
            stopPlayer().then((value) => setState(() {}));
          };
  }

  // ----------------------------------------------------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    Widget _makeBody() {
      return Column(
        children: [
          Container(
            margin: const EdgeInsets.all(3),
            padding: const EdgeInsets.all(3),
            height: 80,
            width: double.infinity,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: const Color(0xFFFAF0E6),
              border: Border.all(
                color: Colors.indigo,
                width: 3,
              ),
            ),
            child: Row(children: [
              ElevatedButton(
                onPressed: getRecorderFn(),
                //color: Colors.white,
                //disabledColor: Colors.grey,
                child: Text(_mRecorder!.isRecording ? 'Stop' : 'Record'),
              ),
              const SizedBox(
                width: 20,
              ),
              Text((_mRecorder!.isRecording
                  ? 'Recording in progress\n[Pos: $pos;  DbLevel: ${((dbLevel * 100.0).floor()) / 100}]'
                  : 'Recorder is stopped')),
            ]),
          ),
          Container(
            margin: const EdgeInsets.all(3),
            padding: const EdgeInsets.all(3),
            height: 80,
            width: double.infinity,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: const Color(0xFFFAF0E6),
              border: Border.all(
                color: Colors.indigo,
                width: 3,
              ),
            ),
            child: Row(children: [
              ElevatedButton(
                onPressed: getPlaybackFn(),
                //color: Colors.white,
                //disabledColor: Colors.grey,
                child: Text(_mPlayer!.isPlaying ? 'Stop' : 'Play'),
              ),
              const SizedBox(
                width: 20,
              ),
              Text(_mPlayer!.isPlaying
                  ? 'Playback in progress'
                  : 'Player is stopped'),
            ]),
          ),
        ],
      );
    }

    return Scaffold(
      body: _makeBody(),
    );
  }

  void handleData(data, EventSink sink) {}
}
