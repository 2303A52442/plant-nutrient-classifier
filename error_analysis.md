# Error Analysis Notes

Healthy -> Nutrient:
- Yellow tint or bright sunlight can make healthy leaves look stressed.
- Background clutter, shadows, and low-quality images are common triggers.
- Some healthy samples have busy scenes with many leaves, stems, or soil in view.

Nutrient -> Healthy:
- Mild chlorosis is missed when yellowing is subtle or spread unevenly.
- Some nutrient samples are blurred or too soft in color contrast.
- The model seems to miss weak but real deficiency patterns when the leaf is only partly yellow.

Main takeaway:
- The failures look like data distribution mismatch, not just a threshold problem.
- Add more realistic healthy variation and clearer nutrient-deficiency examples.