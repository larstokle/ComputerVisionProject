Description:
This is an affine Lucas Kanade template tracker, which performs template tracking between movie frames. For example, to follow cars, moving coronary arteries or measure camera rotation.

The Matlab code is written to show the same steps as in the Literature, not optimized for speed. But also an inverse Lucas Kanada algorithm in c-code for quick template tracking is included, which also contains pixel weighting for more robustness.

Literature:
S. Baker et Al. "Lucas-Kanade 20 Years On: A Unifying Framework"
D. Schreiber, "Robust template tracking with drift correction"

Demo:
Try the TTdemo.m, see screenshot!

Hints:
You can easily adapted the template tracking TTdemo to your own application, for instance:

- Update the template with image data from the new ROI found in the next movie frame.
- Use the ROIs in the next movie frames, also as templates.
- Calculate the speed from difference in affine parameters between movie frames. Speed can be used to detect if tracking fails, or to smooth the tracking, or to predict the next template position.


