// https://pub.dev/packages/pytorch_mobile

import 'package:flutter/material.dart';
import 'package:pytorch_mobile/enums/dtype.dart';
import 'package:pytorch_mobile/model.dart';
import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'package:logger/logger.dart';
//import 'dart:developer' as developer;

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
    final logger = Logger();

    var assetPath = 'assets/models/sample1.pt';
    logger.d("Loading model", assetPath);

    Model customModel = await PyTorchMobile.loadModel(assetPath);

    var tests = [
      [
        0.08522150522041763,
        0.030852370598112397,
        -349.72184795901836,
        77.89076290882933,
        -27.630426193058074,
        20.406152046196976,
        -32.94042322530547,
        6.97672376306317,
        -45.972654643578764,
        4.240683727191109,
        -32.477048890496626,
        4.10894878585488,
        7.191011511169703,
        -25.78896414203323,
        -4.275995403320772,
        4360.281078270551,
        10528.487302421694,
        4076.744587609188
      ],
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
      ]
    ];

    var shape = [1, 18];

    List? prediction =
        await customModel.getPrediction(tests[0], shape, DType.float32);

    logger.d("Log pred, ", prediction);

    List? prediction2 =
        await customModel.getPrediction(tests[1], shape, DType.float32);

    logger.d("Log pred2, ", prediction2);
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
