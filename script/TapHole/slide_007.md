To record those borescope videos, we wrote our own data acquisition program.
First, the rigid wired tip is clamped onto the tap inspection bridge.
Then the scope plugs into a notebook over USB.
Because it follows the standard USB Video Class, Windows treats it like any webcam, with no special driver.
Finally, a short Python program built on OpenCV grabs the frames that feed our detection model.
