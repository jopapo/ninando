// https://pub.dev/packages/pytorch_mobile

import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:pytorch_mobile/enums/dtype.dart';
import 'package:pytorch_mobile/model.dart';
import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'package:sklite/SVM/SVM.dart';
import 'package:sklite/utils/io.dart';
import 'dart:developer' as developer;
import 'package:stats/stats.dart';

/// Example app.
class PredictionWidget extends StatefulWidget {
  @override
  _PredictionWidgetState createState() => _PredictionWidgetState();
}

class _PredictionWidgetState extends State<PredictionWidget> {
  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    super.dispose();
  }

  Future<void> _predict() async {
    //var assetPath = 'assets/models/sample1.pt';
    var assetPath = "assets/models/svc-1.json";
    developer.log("Loading model $assetPath");

    //Model customModel = await PyTorchMobile.loadModel(assetPath);
    //var customModel = loadModel(assetPath);

    loadModel(assetPath).then((x) {
      var svc = SVC.fromMap(json.decode(x));
      predict(svc);
    });
  }

  void predict(SVC customModel) {
    var tests = [
      [
        0.6230733206121143,
        -0.8810694323586448,
        -0.38057744106573466,
        -0.6788806645119531,
        -0.4460772921817446,
        0.14766801504833837,
        -1.7383904143688906,
        -0.01503877115090547,
        -3.2542860444643,
        -0.24428259724341012,
        -3.0374036523756076,
        0.2479474386012638,
        0.8040161036022242,
        -2.891822432879864,
        -0.8535817566023588,
        2.0186658889126603,
        1.4659050440895893,
        0.7048924690172764
      ], // 0
      [
        0.19135959333547403,
        -0.8370162303030986,
        -0.5374184427786426,
        -0.5475556439460879,
        -0.22818162385899296,
        -0.14531519076214672,
        -1.3237762849753938,
        -0.5604561911428055,
        -2.5708854478508294,
        -0.9759990776328569,
        -2.9428700620567607,
        -0.07060079702773044,
        0.9114726820493337,
        -2.723131966435068,
        -1.064646039643554,
        1.2247205427988122,
        0.9827558665928783,
        0.4603174768927352
      ] // 1
    ];

    //var shape = [18];
    for (var i = 0; i < tests.length; i++) {
      // var prediction =
      //     await customModel.getPrediction(tests[i], shape, DType.float32);

      var prediction = customModel.predict(tests[i]);
      developer.log("Log pred: ${[i, prediction]}");
    }
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
        color: const Color(0xFFFAF0E6),
        border: Border.all(
          color: Colors.indigo,
          width: 3,
        ),
      ),
      child: ElevatedButton(
        onPressed: _predict,
        //color: Colors.white,
        //disabledColor: Colors.grey,
        child: const Text('Predict'),
      ),
      //const Text('B'),
    );
  }
}
