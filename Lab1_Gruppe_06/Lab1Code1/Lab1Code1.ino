#include <Wire.h>
#include "SparkFun_MMA8452Q.h"

MMA8452Q accel;

unsigned long startTime;

void setup()
{
  Serial.begin(57600);
  Wire.begin();

  if (accel.begin() == false)
  {
    Serial.println(
      "Not Connected. Please check connections and read the hookup guide."
    );

    while (1);
  }

  accel.setScale(SCALE_4G);
  accel.setDataRate(ODR_100);

  // Startzeit der Datenerfassung speichern
  startTime = millis();

  // Spaltenüberschriften für die TXT-Datei
  Serial.println("Time\tX\tY\tZ");
}

void loop()
{
  if (accel.available())
  {
    // Zeit seit Beginn der Datenerfassung in Millisekunden
    unsigned long timestamp = millis() - startTime;

    // Messwerte einlesen
    float x = accel.getX();
    float y = accel.getY();
    float z = accel.getZ();

    // Ausgabe: Zeit, X, Y, Z
    Serial.print(timestamp);
    Serial.print("\t");

    Serial.print(x);
    Serial.print("\t");

    Serial.print(y);
    Serial.print("\t");

    Serial.print(z);
    Serial.println();
  }
}