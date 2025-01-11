#define pan 18
#define tilt 19
#include <ESP32Servo.h>

Servo servoPan;
Servo servoTilt;
String inputString;

void setup() {
    Serial.begin(9600);
    servoPan.attach(pan);
    servoTilt.attach(tilt);

    servoPan.write(90);
    servoTilt.write(90);
}

void loop(){
    while (Serial.available() > 0){
        inputString = Serial.readStringUntil('\r');
        Serial.println(inputString);
        int x_axis = inputString.substring(0, inputString.indexOf(',')).toInt();
        int y_axis = inputString.substring(inputString.indexOf(',') + 1).toInt();

        int x = map(x_axis, 0, 640, 180, 0);
        int y = map(y_axis, 0, 480, 180, 0);

        servoPan.write(x);
        servoTilt.write(y);
    }
}