# Literature Review

This review surveys the foundational and applied literature underpinning the color detection, visual tracking, and MAVLink actuation pipeline implemented in drone-seeker. The works are grouped by topic area and discussed in the context of the design decisions made in this system.

---

## 1. Color Space Selection — HSV

The choice of the HSV (Hue–Saturation–Value) color space as the basis for detection stems directly from the formal framework introduced by Smith [1]. By separating chromatic information (Hue) from luminance (Value), HSV decouples the color identity of an object from the ambient illumination falling on it. This is critical in outdoor drone applications where sunlight intensity and angle vary continuously during a flight. The geometric and perceptual properties of HSV are covered comprehensively by Gonzalez and Woods [2], whose treatment of color segmentation remains the standard engineering reference. In drone-seeker, all three detection methods operate in HSV: the Gaussian back-projection method builds a confidence histogram over the hue channel; the adaptive method normalizes and thresholds the hue channel locally; and the dual inRange method applies per-channel bounds directly in HSV space.

---

## 2. Histogram Back-Projection

The Gaussian back-projection detector in drone-seeker is a direct application of the color indexing framework introduced by Swain and Ballard [3]. Their seminal 1991 paper showed that a color histogram of a reference object could be used to compute, for each pixel of a query image, the probability that the pixel belongs to that object — a process they termed back-projection. The resulting probability map is then thresholded to obtain a detection mask. In drone-seeker, `cv2.calcBackProject` is called with a pre-built Gaussian confidence histogram (`_confidence_hist`) that assigns high probability only to hue bins within ±2.5σ of the calibrated mean hue, suppressing all other hues to zero. The Saturation and Value constraints are enforced separately to reject achromatic and dark regions.

---

## 3. Circular Statistics for Hue Modelling

Hue is a circular (periodic) quantity: the hue of a "hot pink" object may be centered near H = 165 in OpenCV's 0–179 scale, but when pink spans the boundary (e.g., H = 175 wraps around to H = 0), standard Euclidean mean and standard deviation give incorrect results. The correct treatment is provided by Fisher [4] and Mardia and Jupp [5], whose textbooks establish the theory of circular statistics. The mean of a circular distribution is computed by mapping each angle to a unit vector, averaging the vectors, and taking the argument of the resultant — a method that handles wrap-around naturally. The circular standard deviation (or angular deviation) is derived from the magnitude of that resultant. Drone-seeker's `_fit_gaussian` function implements this directly for hue calibration.

The theoretical necessity for circular arithmetic on Hue is further demonstrated by Hanbury and Serra [6], who show that morphological operators and distance metrics applied naively to the Hue channel — treating it as a linear variable — produce topologically inconsistent results. Their work motivates the use of circular distance in `_confidence_hist` construction and in the angular wrapping applied during adaptive thresholding.

---

## 4. Adaptive Thresholding

The adaptive detection method in drone-seeker was motivated by the limitations of global thresholding under non-uniform illumination. Sauvola and Pietikäinen [7] established the principle of local adaptive binarization: the threshold at each pixel is derived from the local mean and standard deviation within a spatial neighbourhood, making the decision invariant to slow spatial gradients in illumination. Bradley and Roth [8] reformulated this approach using integral images, achieving near-constant-time computation regardless of neighbourhood size — a key property for real-time video processing. In drone-seeker's `_mask_adaptive`, the Hue channel is first normalized to the local illumination level, then `cv2.adaptiveThreshold` (which uses a local mean threshold equivalent to Bradley–Roth) is applied with a block size of 21 pixels. The result is masked against a coarse hue gate to retain only hue-plausible regions.

---

## 5. CamShift Tracking

Once a target is detected through five consecutive frames, drone-seeker transitions from detection to tracking using the CamShift algorithm. CamShift (Continuously Adaptive Mean Shift) was introduced by Bradski [9, 10] as an extension of the Mean Shift algorithm to video sequences. Mean Shift is a non-parametric mode-seeking procedure that iteratively shifts a window toward the local maximum of a probability density function — in this case, a back-projected color histogram. CamShift additionally adapts the size and orientation of the tracking window at each frame to match the current scale of the tracked object. The mathematical foundation of this approach, using kernel density estimation with color histograms as the underlying probability model, was rigorously formalized by Comaniciu, Ramesh, and Meer [11], whose kernel-based tracker provides the theoretical basis for the practical CamShift implementation available as `cv2.CamShift` in OpenCV.

In drone-seeker, CamShift is initialized with the bounding rectangle returned by the last successful detection step. The tracker runs on the same Gaussian confidence back-projection map used for detection, ensuring consistent color modeling between the two stages. If the tracker window collapses below a minimum area or the confidence map loses signal, the system falls back to detection mode.

---

## 6. Mathematical Morphology

Raw thresholded binary masks from any of the three detection methods typically contain salt-and-pepper noise (isolated positive pixels from accidental hue matches) and small intra-object gaps (pixels that fail the threshold near edges or in shadowed regions). Mathematical morphology, whose formal foundations were established by Serra [12], provides the standard operations for cleaning these artefacts. An Opening (erosion followed by dilation with the same structuring element) removes isolated noise without significantly eroding large connected regions. A subsequent Dilation expands the surviving regions to fill small holes and merge nearby fragments. Drone-seeker applies this OPEN → DILATE sequence after the majority-vote fusion of the three detection masks, using a 5×5 elliptical structuring element for both operations.

---

## 7. MAVLink Communication Protocol

The tracking error signals computed by drone-seeker are transmitted to the ArduPlane flight controller over a serial link using the MAVLink protocol. MAVLink was introduced by Lorenz Meier at ETH Zurich in 2009 and is now maintained by the Dronecode Project [14]. It is a lightweight framing protocol designed for resource-constrained embedded systems: each message carries a 1-byte CRC extra (a message-type-specific constant mixed into the checksum) to detect version mismatches and message corruption without a separate integrity field. The protocol architecture, message framing, and CRC-extra mechanism are surveyed comprehensively by Koubaa et al. [13], who also review the integration of MAVLink into both ArduPilot and PX4 autopilots.

In drone-seeker, tracking errors are transmitted as `DEBUG_VECT` messages (MAVLink ID 250), a standard message present in the compiled `MAVLINK_MESSAGE_CRCS` table of ArduPlane. This choice was driven by a concrete failure mode: MAVLink's `mavlink_get_msg_entry()` silently discards any message whose ID is not in the compiled table, regardless of content. Messages sent with non-standard IDs (IDs 229, 230, 202 were all attempted) were dropped at the receiver before reaching the application layer. Switching to the standard `DEBUG_VECT` message resolved the issue and enabled reliable reception of tracking errors in the `ModeTracking` flight mode handler.
