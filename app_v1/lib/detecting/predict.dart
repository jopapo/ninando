// https://pub.dev/packages/pytorch_mobile

import 'dart:convert';
import 'dart:math';

import 'package:flutter/material.dart';
// import 'package:pytorch_mobile/enums/dtype.dart';
// import 'package:pytorch_mobile/model.dart';
// import 'package:pytorch_mobile/pytorch_mobile.dart';
// ignore: import_of_legacy_library_into_null_safe
import 'package:sklite/SVM/SVM.dart';
// ignore: import_of_legacy_library_into_null_safe
import 'package:sklite/utils/io.dart';
import 'dart:developer' as developer;

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

  /// https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
  List<double> standardScaler(List<double> values) {
    var featureMeans = [
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

    var featureVars = [
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

    var result = List<double>.generate(
        values.length,
        (index) =>
            (values[index] - featureMeans[index]) / sqrt(featureVars[index]));
    (values.length);
    developer.log("standardScaler $result");
    return result;
  }

  void predict(SVC customModel) {
    var tests = [
      [
        0.07456768416473318,
        0.0349972660301076,
        -345.31093390303255,
        83.49781324028139,
        -28.590596779109156,
        20.736835578132396,
        -33.868432888437034,
        3.2073644044919245,
        -45.18277865385512,
        0.24046769361346615,
        -34.63165851314217,
        5.013043599034684,
        7.490170033393631,
        -24.33343967795372,
        -6.81495497802709,
        4115.341337344389,
        10287.12557551479,
        4037.44639117785
      ], // 0
      [
        0.0575254223462877,
        0.03878751955332187,
        -359.36640204409713,
        89.16152922681202,
        -19.130301450286275,
        16.12922362087774,
        -26.75561651732695,
        -5.065895164821928,
        -35.52834761446977,
        -8.743458662674765,
        -33.517937830595294,
        1.7380714627402012,
        8.499837581284362,
        -22.920507246961172,
        -8.528495653668578,
        3445.647021016951,
        9135.324032953886,
        3751.9222202377164
      ] // 1
    ];

    //var shape = [18];
    for (var i = 0; i < tests.length; i++) {
      // var prediction =
      //     await customModel.getPrediction(tests[i], shape, DType.float32);

      var stdVals = standardScaler(tests[i]);

      var prediction = customModel.predict(stdVals);
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