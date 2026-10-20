Beyond the detector, we count threads with simple line sampling on a grayscale image.
The challenge is that threads appear as alternating light and dark rings in a circular pattern.
First, we draw a line from the right edge to the center, so it crosses every ring.
Then we read the grayscale intensity at each point along that line.
Finally, we count the intensity drops, since each dark dip in a real borescope image is one groove.
