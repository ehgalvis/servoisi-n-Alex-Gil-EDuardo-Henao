/*
  # This Sample code is to test the Digital Servo Shield.
  # Editor : Leff, original by YouYou@DFRobot(Version1.0)
  # Date   : 2016-1-19
  # Ver    : 1.1
  # Product: Digital Servo Shield for Arduino
  # Hardwares:
  1. Arduino UNO
  2. Digital Servo Shield for Arduino
  3. Digital Servos( Compatible with AX-12,CDS55xx...etc)
  4. Power supply:6.5 - 12V
  # How to use:
  If you don't know your Servo ID number, please
  1. Open the serial monitor, and choose NewLine,115200
  2. Send command:'d',when it's finished, please close the monitor and re-open it
  3. Send the command according to the function //controlServo()//
*/

#include <SPI.h>
#include <ServoCds55.h>
ServoCds55 myservo;

int velocidad = 30;
int servoNum = 1;

void setup()
{
  Serial.begin (115200);
  Serial.print ("Ready...\n");
  myservo.begin ();
  myservo.Reset(servoNum);
  myservo.setVelocity(velocidad);
}

void loop()
{
}

void serialEvent()
{
  while (Serial.available() > 0)
  {
    //char inChar = Serial.read();
    char inChar = Serial.read();
    
    if (inChar == 'i')
    {
      // convert the incoming byte to a char and add it to the string:
      myservo.rotate(servoNum, velocidad); //   Anti CW    ID:1  Velocity: 150_middle velocity  300_max
    }
    else if (inChar == 'd')
    {
      myservo.rotate(servoNum, -velocidad); //   Anti CW    ID:1  Velocity: 150_middle velocity  300_max
    }
    else if (inChar == 'q')
    {
      // No gire si se pierde una linea
      myservo.rotate(servoNum, 0); //   Anti CW    ID:1  Velocity: 150_middle velocity  300_max
    }
  }
  Serial.flush();
}

