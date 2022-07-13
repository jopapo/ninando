// URL: https://stackoverflow.com/questions/45523370/how-to-make-a-phone-call-from-a-flutter-app

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_phone_direct_caller/flutter_phone_direct_caller.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:developer' as developer;

/// Example app.
class CallNotificatorWidget extends StatefulWidget {
  final StreamController<bool> onAlertThreashold;

  const CallNotificatorWidget({Key? key, required this.onAlertThreashold})
      : super(key: key);

  @override
  _CallNotificatorWidgetState createState() => _CallNotificatorWidgetState();
}

class _CallNotificatorWidgetState extends State<CallNotificatorWidget> {
  late TextEditingController _phoneNumberController;

  @override
  void initState() {
    super.initState();

    _phoneNumberController =
        TextEditingController.fromValue(const TextEditingValue(text: ''));

    SharedPreferences.getInstance().then((instance) {
      var savedContactPhone = instance.getString("contact-phone");
      _phoneNumberController.text = savedContactPhone ?? '';
    });

    widget.onAlertThreashold.stream.listen((isAlerted) {
      if (isAlerted) {
        _makeCall();
      }
    });
  }

  void _contactPhoneChanged(String value) {
    SharedPreferences.getInstance()
        .then((instance) => instance.setString("contact-phone", value));
  }

  @override
  void dispose() {
    _phoneNumberController.dispose();

    super.dispose();
  }

  Future<bool?> _makeCall() {
    String phoneNumber = _phoneNumberController.text;
    return FlutterPhoneDirectCaller.callNumber(phoneNumber);
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
            onChanged: _contactPhoneChanged,
          )),
    );
  }
}
