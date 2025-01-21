#include <ESP32Servo.h>
#define panPin 18
#define tiltPin 19

// Define the PID Controller class
class PIDController {
  private:
    float Kp, Ki, Kd;
    float setpoint;
    float previous_error;
    float integral;

  public:
    PIDController(float Kp, float Ki, float Kd, float setpoint) {
      this->Kp = Kp;
      this->Ki = Ki;
      this->Kd = Kd;
      this->setpoint = setpoint;
      this->previous_error = 0;
      this->integral = 0;
    }

    float compute(float process_variable, float dt) {
      // Calculate error
      float error = setpoint - process_variable;

      // Proportional term
      float P_out = Kp * error;

      // Integral term
      integral += error * dt;
      float I_out = Ki * integral;

      // Derivative term
      float derivative = (error - previous_error) / dt;
      float D_out = Kd * derivative;

      // Compute total output
      float output = P_out + I_out + D_out;

      // Update previous error
      previous_error = error;

      return output;
    }

    void setSetpoint(float new_setpoint) {
      setpoint = new_setpoint;
    }
};

// Initialize servos and PID controllers
Servo panServo;
Servo tiltServo;

float pan_setpoint = 320; // Center of 640px width
float tilt_setpoint = 240; // Center of 480px height
PIDController pan_pid(5.0, 0.1, 0.05, pan_setpoint);
PIDController tilt_pid(5.0, 0.1, 0.05, tilt_setpoint);

float current_pan_angle = 90; // Start at center position
float current_tilt_angle = 90;
unsigned long previous_time = 0;
float dt = 0.01;
unsigned long last_detection_time = 0;
const unsigned long detection_timeout = 3000; // 3 seconds timeout

void setup() {
  Serial.begin(115200);

  // Attach servos
  panServo.attach(panPin);  // Adjust to your pan servo pin
  tiltServo.attach(tiltPin); // Adjust to your tilt servo pin

  // Set servos to initial positions
  panServo.write(current_pan_angle);
  tiltServo.write(current_tilt_angle);
}

void loop() {
  unsigned long current_time = millis();
  dt = (current_time - previous_time) / 1000.0; // Convert to seconds
  if (dt <= 0) return;

  // Check for incoming data from the computer
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    int commaIndex = input.indexOf(',');
    if (commaIndex > 0) {
      // Parse coordinates from the input string
      float object_x = input.substring(0, commaIndex).toFloat();
      float object_y = input.substring(commaIndex + 1).toFloat();

      // Compute PID outputs
      float pan_output = pan_pid.compute(object_x, dt);
      float tilt_output = tilt_pid.compute(object_y, dt);

      // Update servo positions
      current_pan_angle = constrain(current_pan_angle - pan_output * dt, 0, 180);
      current_tilt_angle = constrain(current_tilt_angle - tilt_output * dt, 0, 180);

      panServo.write(current_pan_angle);
      tiltServo.write(current_tilt_angle);

      // Update last detection time
      last_detection_time = current_time;

      // Debug output
      Serial.print("Object X: ");
      Serial.print(object_x);
      Serial.print(", Object Y: ");
      Serial.print(object_y);
      Serial.print(", Pan Angle: ");
      Serial.print(current_pan_angle);
      Serial.print(", Tilt Angle: ");
      Serial.println(current_tilt_angle);
    }
  }

  // Check for detection timeout
  if (current_time - last_detection_time > detection_timeout) {
    // Return to home position if no detection
    current_pan_angle = 90;
    current_tilt_angle = 90;
    panServo.write(current_pan_angle);
    tiltServo.write(current_tilt_angle);
  }

  // Update previous time
  previous_time = current_time;
}
