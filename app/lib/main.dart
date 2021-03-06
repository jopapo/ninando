import 'package:flutter/material.dart';

import 'recording/record_to_stream.dart';
import 'contacting/call.dart';
import 'detecting/predict.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      //debugShowCheckedModeBanner: false,
      title: 'Ninando',
      theme: ThemeData(
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        brightness: Brightness.dark,
      ),
      themeMode: ThemeMode.dark,
      home: const MyHomePage(title: 'Ninando Home'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  Widget build(BuildContext context) {
    var predictor = PredictionWidget();
    var recorder = RecordToStream(
        onNewRecordingSubscription: predictor.onNewRecordingSubscription);

    return Scaffold(
      //backgroundColor: Colors.black87,
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Column(
        //mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[
          SizedBox(
            height: 80,
            child: recorder,
          ),
          SizedBox(
            height: 80,
            child: predictor,
          ),
          Expanded(
            child: CallNotificatorWidget(
                onAlertThreashold: predictor.onAlertThreashold,
                onRecordToggle: recorder.onRecordToggle),
          ),
        ],
      ),
    );
  }
}
