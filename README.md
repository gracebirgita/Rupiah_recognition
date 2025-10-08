# Rupiah Recognition

Rupiah Recognition is a computer vision + speech project that detects Indonesian Rupiah banknotes in images using YOLOv8, extracts the denomination, and then speaks out the total value in natural-language form using a T5 based speech synthesis model.
This project is intended to showcase end-to-end multimodal capabilities object detection, text interpretation, and speech generation

- Automatically recognizing and vocalizing the value of currency notes is useful for accessibility applications (e.g. for the visually impaired), automated teller machines, retail automation, or finance auditing tools.

### Objectives:
- Detect and localize Rupiah banknotes in arbitrary images.
- Recognize the denomination (e.g. “Rp 10000”, “Rp 50000”).
- Generate a spoken sentence like “lima puluh ribu rupiah.”
- Demonstrate integration of detection, vision-language, and speech modules in a unified pipeline.

## The pipeline:
1. **Detection stage**: YOLOv8 detects banknote(s) in the image and outputs bounding boxes with class labels corresponding to denominations.
2. **Extraction stage**: From the detections, we select the predicted label(s) (e.g. “10000”) and convert them into textual form (e.g. “sepuluh ribu”).
3. **Speech stage**: A T5-based or T5-conditioned text-to-speech (TTS) model synthesizes an audio utterance that reads out the recognized denomination in natural language.
