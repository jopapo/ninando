// URL: https://stackoverflow.com/questions/45523370/how-to-make-a-phone-call-from-a-flutter-app

//import 'dart:async';
//import 'dart:io';
import 'package:flutter/material.dart';
//import 'package:flutter_sound/flutter_sound.dart';
//import 'package:path_provider/path_provider.dart';
//import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_phone_direct_caller/flutter_phone_direct_caller.dart';

/// Example app.
class CallNotificatorWidget extends StatefulWidget {
  @override
  _CallNotificatorWidgetState createState() => _CallNotificatorWidgetState();
}

class _CallNotificatorWidgetState extends State<CallNotificatorWidget> {
  late TextEditingController _phoneNumberController;

  @override
  void initState() {
    super.initState();

    _phoneNumberController = TextEditingController.fromValue(
      const TextEditingValue(
        text: '+5547988026050',
      ),
    );
  }

  @override
  void dispose() {
    _phoneNumberController.dispose();

    super.dispose();
  }

  Future<void> _makeCall() async {
    String phoneNumber = _phoneNumberController.text;
    await FlutterPhoneDirectCaller.callNumber(phoneNumber);
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
      child: ConstrainedBox(
          constraints: BoxConstraints.tight(const Size(200, 50)),
          child: TextFormField(
            decoration: InputDecoration(
              //border: InputBorder.none,
              suffixIcon: IconButton(
                icon: const Icon(Icons.call),
                onPressed: _makeCall,
                color: Colors.blue,
              ),
              labelText: 'Phone to call:',
            ),
            //autofocus: true,
            keyboardType: TextInputType.phone,
            //maxLines: null,
            controller: _phoneNumberController,
          )),
      //const Text('B'),
    );
  }
}
