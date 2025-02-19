## Description
This repository provides a step-by-step guide to converting segmentation masks generated using YOLOv11 into an annotation format compatible with the CVAT server. The code can be adapted for other models as well.

## Use Case
I need to annotate videos with the label "road."
To achieve this, I generate pseudo-annotations by fine-tuning a segmentation model on an existing dataset and then refine these annotations using CVAT software.

## Steps
### Step 0: Upload the Video to CVAT
- Upload a video to CVAT and edit the task settings (constructor) by adding the labels of interest (in this case, "**road**").
![alt text](cvat_constructor.png)

### Step 1: Export the Annotation XML
- Download the XML annotation file for the video by exporting the task.
- Use the "**CVAT for video 1.1**" export format.
![alt text](export_xml.png)

### Step 2: Generate and Save Pseudo-Annotations
- Use `yolov11_infer_masks_to_poly.py` to generate pseudo-annotations and integrate them into the exported XML.
- This script takes the video and its corresponding XML file (from Step 1) as input, infers the segmentation mask and polyline for the "**road**" class using a fine-tuned YOLOv11 segmentation model, and updates the XML file accordingly.
- The updated annotation files (`.xml`) will be saved in the `output_xml_annots_with_road_annotation` folder.