import 'package:flutter_test/flutter_test.dart';

import 'package:phone_collector/main.dart';

void main() {
  testWidgets('shows no camera message when cameras are unavailable',
      (WidgetTester tester) async {
    await tester.pumpWidget(const CollectorApp(cameras: []));
    await tester.pump();

    expect(find.text('No camera found on this phone.'), findsOneWidget);
  });
}
