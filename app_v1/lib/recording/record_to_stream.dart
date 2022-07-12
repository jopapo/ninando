import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';

const int tSampleRate = 44100;
const int tBitRate = 16000;
typedef _Fn = void Function();

/// Recorder Widget
class RecordToStream extends StatefulWidget {
  Future<StreamSubscription<Food>> Function(StreamController<Food>, int)
      onNewRecordingSubscription;

  RecordToStream({Key? key, required this.onNewRecordingSubscription})
      : super(key: key);

  @override
  _RecordToStreamState createState() => _RecordToStreamState();
}

class _RecordToStreamState extends State<RecordToStream> {
  FlutterSoundRecorder? _mRecorder = FlutterSoundRecorder();
  bool _mRecorderIsInited = false;
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
    super.initState();

    _openRecorder();
  }

  @override
  void dispose() {
    stopRecorder();
    _mRecorder!.closeAudioSession();
    _mRecorder = null;

    _mRecordingDataSubscription?.cancel();
    _mRecordingDataSubscription = null;
    _mRecordingProgressSubscription?.cancel();
    _mRecordingProgressSubscription = null;

    super.dispose();
  }

  Future<void> record() async {
    assert(_mRecorderIsInited);

    var recordingDataController = StreamController<Food>();

    _mRecordingDataSubscription = await widget.onNewRecordingSubscription
        .call(recordingDataController, tSampleRate);

    await _mRecorder!.startRecorder(
        toStream: recordingDataController.sink,
        codec: Codec.pcm16,
        numChannels: 1, // Mono
        sampleRate: tSampleRate,
        bitRate: tBitRate);

    setState(() {});
  }

  Future<void> stopRecorder() async {
    await _mRecorder!.stopRecorder();

    await _mRecordingDataSubscription?.cancel();
    _mRecordingDataSubscription = null;
  }

  _Fn? getRecorderFn() {
    if (!_mRecorderIsInited) {
      return null;
    }
    return _mRecorder!.isStopped
        ? record
        : () {
            stopRecorder().then((value) => setState(() {}));
          };
  }

  @override
  Widget build(BuildContext context) {
    Widget _makeBody() {
      return Container(
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
      );
    }

    return Scaffold(
      body: _makeBody(),
    );
  }

  void handleData(data, EventSink sink) {}
}
