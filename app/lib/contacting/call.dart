// URL: https://stackoverflow.com/questions/45523370/how-to-make-a-phone-call-from-a-flutter-app

import 'dart:async';

import 'package:intl/intl.dart';
import 'package:flutter/material.dart';
import 'package:flutter_phone_direct_caller/flutter_phone_direct_caller.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Example app.
class CallNotificatorWidget extends StatefulWidget {
  final StreamController<bool> onAlertThreashold;

  const CallNotificatorWidget({Key? key, required this.onAlertThreashold})
      : super(key: key);

  @override
  _CallNotificatorWidgetState createState() => _CallNotificatorWidgetState();
}

class _CallNotificatorWidgetState extends State<CallNotificatorWidget> {
  late final TextEditingController _phoneNumberController;
  final List<String> _history = <String>[];
  late final Future<SharedPreferences> _prefs;

  @override
  void initState() {
    super.initState();

    _phoneNumberController =
        TextEditingController.fromValue(const TextEditingValue(text: ''));

    _prefs = SharedPreferences.getInstance();

    _prefs.then((instance) {
      var savedContactPhone = instance.getString("contact-phone");
      _phoneNumberController.text = savedContactPhone ?? '';

      var storedHistory = instance.getStringList("history");
      if (storedHistory != null && storedHistory.isNotEmpty) {
        setState(() {
          _history.addAll(storedHistory);
        });
      } else {
        history(add: "Primeira execução.");
      }
    });

    widget.onAlertThreashold.stream.listen((isAlerted) {
      if (isAlerted) {
        history(add: "Choro detectado.");
        _makeCall()
            .then((value) => history(
                complement:
                    value.isEmpty ? "Sem notif." : "Ligando para $value!"))
            .catchError(
                (onError) => history(complement: "Erro notif.: $onError!"));
      } else {
        history(add: "Choro parou.");
      }
    });
  }

  void _contactPhoneChanged(String value) {
    _prefs.then((instance) {
      instance.setString("contact-phone", value);
    });
  }

  @override
  void dispose() {
    _phoneNumberController.dispose();

    super.dispose();
  }

  Future<String> _makeCall() async {
    String phoneNumber = _phoneNumberController.text;
    if (phoneNumber.isNotEmpty) {
      return FlutterPhoneDirectCaller.callNumber(phoneNumber)
          .then((value) => value == true ? phoneNumber : '');
    }
    return '';
  }

  void history({String? add, String? complement}) {
    var dateFormat = DateFormat('yyyy-MM-dd HH:mm:ss');
    setState(() {
      if (add != null) {
        _history.insert(0, "${dateFormat.format(DateTime.now())} - $add");
      }
      if (complement != null) {
        _history.first += " $complement";
      }
    });
    _prefs.then((instance) => instance.setStringList("history", _history));
  }

  Widget buildList(BuildContext context) {
    return SelectableText(_history.join("\n"),
        toolbarOptions: const ToolbarOptions(copy: true, selectAll: true),
        showCursor: true,
        textAlign: TextAlign.left,
        maxLines: 15);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _body(),
    );
  }

  Widget _body() {
    return Column(children: [
      Container(
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
        child: ConstrainedBox(
            constraints: BoxConstraints.tight(const Size(200, 50)),
            child: TextFormField(
              decoration: InputDecoration(
                //border: InputBorder.none,
                suffixIcon: IconButton(
                  icon: const Icon(Icons.call),
                  onPressed: _makeCall,
                  color: Colors.indigo,
                ),
                labelText: 'Phone to call:',
              ),
              //autofocus: true,
              keyboardType: TextInputType.phone,
              //maxLines: null,
              controller: _phoneNumberController,
              onChanged: _contactPhoneChanged,
            )),
      ),
      Column(children: [
        Container(
            margin: const EdgeInsets.all(3),
            padding: const EdgeInsets.all(3),
            width: double.infinity,
            decoration: BoxDecoration(
              border: Border.all(
                color: Colors.indigo,
                width: 3,
              ),
            ),
            child: buildList(context))
      ]),
    ]);
  }
}
