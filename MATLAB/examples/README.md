To run the examples, in the MATLAB command window run `runExamples('all')`


I played around with only incrementing a single bin and incrementing a 3x3 grid of bins
while still preserving the center of the line as the peak value(s). I ended up sticking 
with incrementing the single bin. This scheme was a lot quicker than the 3x3 grid and I was
able to keep the noise down more as well.

To find the peaks, I just looped through the matrix and found values that were above my threshold.
To set the threshold, I set them by hand for each image so that each one found the lines I thought
it should find and didn't have too much noise/duplicates.

I ended up using the edge image to help me determine the line segments.
I basically plotted each point of the line on the image and checked to make sure
there was an edge within some threshold distance from the point.
From that, I was able to get the start and stop of the line segment.
