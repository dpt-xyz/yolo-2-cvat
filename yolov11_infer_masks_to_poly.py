'''
Input: Videos and corresponding XML files (e.g., abc.mp4 and abc.xml)
Output: updated xml that has mask nnotations (verify by uploading on CVAT)
'''

from ultralytics import YOLO

import os
import gc
import cv2
import xml.etree.ElementTree as ET
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------------------------------

def process_frame_and_predict_masks_road(model, results): # OpenCV -> cv2.imread('image.jpg')
    
    # Define folder to save masks
    # save_folder = "masks_output"
    # os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist
        
    # Process results generator
    for result in results:
        # boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs (YOLOv11 format)
        # print("masks.orig_shape", masks.orig_shape)
        # print("masks.xy", masks.xy) # list of arrys with poly points
        class_ids = result.boxes.cls.tolist() # List of detected class IDs
        class_names = model.names  # Dictionary mapping class index to label
        target_class = 5.0  # road # Change this based on your dataset
        
        # Extract masks and polys for the target class
        target_masks = [masks.data[i] for i in range(len(class_ids)) if class_ids[i] == target_class]
        target_poly = [masks.xy[i] for i in range(len(class_ids)) if class_ids[i] == target_class]

        # Print number of detected instances for the target class
        # print(f"Detected {len(target_masks)} instances of '{class_names[target_class]}'")

        # Print each mask separately
        # for idx, mask in enumerate(target_masks):
            # print(f"Mask {idx + 1} for {class_names[target_class]}:\n", mask)
            # mask_numpy = mask.cpu().numpy()  # Convert tensor to NumPy array
            # save_path = os.path.join(save_folder, f"{class_names[target_class]}_{idx+1}.png")  # Define filename
            # plt.imsave(save_path, mask_numpy, cmap="gray") # Save the mask as a grayscale image
            # print(f"Saved mask {idx+1} for '{class_names[target_class]}' at {save_path}")
       
        # print(f"target_poly type: {type(target_poly)} and target_poly items: {len(target_poly)}")
        return target_poly

# -----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Load model
    model = YOLO("./best.pt")  # load a custom model

    video_folder = "/original_videos"
    xml_folder = "/xml_annots"
    # output_vid_folder = "traffic_vid_and_annot/output_vids_with_masks/"
    output_xml_folder = "/output_xml_annots_with_road_annotation/"
    # Get a list of all video files in the video folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    video_files = sorted(video_files) # Sort the video files alphabetically
    # video_files = ['20211109123408_0060.mp4']

    # Iterate over each video file
    for video_file in video_files:
        print(f"Processing: {video_file}")
        xml_file = os.path.join(xml_folder, os.path.splitext(video_file)[0] + ".xml") # Construct the corresponding XML file path
        print("xml_file ", xml_file)
        # exit(0)

        # Check if XML file already exists
        output_xml_file = os.path.join(output_xml_folder, os.path.splitext(video_file)[0] + ".xml")
        if os.path.exists(output_xml_file):
            print(f"OUTPUT XML FILE ALREADY EXISTS for {video_file} in the folder: {output_xml_folder}\nskipping...")
            continue
            
        # Your processing code here for videos without existing XML files
        print(f"No existing XML found for {video_file}, processing...")

        # Construct the full paths to the video and output folders
        video_file_path = os.path.join(video_folder, video_file)

        # Create the output folder if it doesn't exist
        os.makedirs(output_xml_folder, exist_ok=True)

        # Parse the XML data from the file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Read the video file
        cap = cv2.VideoCapture(video_file_path)

        # video writer
        while(cap.isOpened()):
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fwidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
            fheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

            frames = []
            road_polys = []
            
            ctr = 0
            # Iterate over all frames in the video
            for frame_number in range(frame_count):
                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    break
                if(frame_number % 5 != 0):  # in our videos uploaded to CVAT, the frame step is 5 i.e., only every 5th frame is uploaded and therefore annotations are generated for every 5th frame
                    continue
                frames.append(frame)
                print(f"frame_number: {frame_number}")
                ctr = ctr+1
                print("Frame being processed: ", ctr)

                # Process motorcycle masks and get their track IDs
                results = model(frame, stream=False, save=False, conf=0.55, device=0, classes=[5, 3, 4], name="without_TTA")  # generator of Results objects

                target_poly_road_frame = process_frame_and_predict_masks_road(model, results) # target_poly_road_frame is "list" of arrays, where each array is the mask of road
                road_polys.append(target_poly_road_frame)


            # fetch xml file structured_annotations/175.xml
            tree = ET.parse(xml_file)
            root = tree.getroot()

            frame_height = fheight
            frame_width = fwidth

            # search for original_size tag in meta/project/tasks/task/original_size
            for original_size in root.findall('meta/project/tasks/task/original_size'):
                for child in original_size:
                    if child.tag == 'height':
                        frame_height = int(child.text)
                    if child.tag == 'width':
                        frame_width = int(child.text)

            # get polygons for each frame
            all_polygons_road = []

            # print("motor_track_ids: ", motor_track_ids)
            # 1/0
            for i, frame in enumerate(frames): # there are 300 frames in our video (after considering a frame step size of 5), so this loop goes on for 300 times
                
                p_r = road_polys[i] # p_r should be an array of lists -> p_r = array([[a,b],[c,d],[e,f]...]) where each list depicts the corrdinates of a polygon point.
                p_r = [p_r]
                # But here p_r = [[array([[a,b],[c,d],[e,f]...])]]
                # print(f"p_r: {p_r}\np_c: {p_c}\n")
                # exit(0)

                if(len(p_r) == 0):
                    p_r = []

                all_polygons_road.append(p_r)

            # print("len(all_polygons_road): ", len(all_polygons_road))
            # print(all_polygons_road)

            all_polygons_road_copy = all_polygons_road.copy()

            # plot polygons for each frame
            # Polygon Coordinate Scaling:
            # The nested loops handle different levels of the polygon data structure:
            for i in range(len(frames)): # loop through each frame - 300 of them
                # resize the frame and all the polygons to 1920xframe_height
                frames[i] = cv2.resize(frames[i], (frame_width, frame_height))
                # cv2.imwrite("/frameCheck_"+str(i)+".png", frames[i])
                # print(f"all_polygons_road_copy[i] -> {len(all_polygons_road_copy[i])}")
                
                for j in range(len(all_polygons_road_copy[i])): # j=1 as all_polygons_road_copy[i] is of form [[ array([[x,y],[a,b]]), array([[x,y],[a,b]]), array([[x,y],[a,b]]) ]]
                    # print(f"all_polygons_road_copy[i=1, j=1]\n", all_polygons_road_copy[i]) # all_polygons_road_copy[i] is of the form mentioned in above comment but I need to see if for multiple such arrays, it will be of the form mentioned above
                    # exit(0)

                    for each_road_polygon in range(len(all_polygons_road_copy[i][j])):
                        print("len(all_polygons_road_copy[i][j]) ", len(all_polygons_road_copy[i][j]))
                        for k in range(len(all_polygons_road_copy[i][j][each_road_polygon])):
                            all_polygons_road_copy[i][j][each_road_polygon][k][0] = float(min(all_polygons_road_copy[i][j][each_road_polygon][k][0], frame_width))
                            all_polygons_road_copy[i][j][each_road_polygon][k][1] = float(min(all_polygons_road_copy[i][j][each_road_polygon][k][1], frame_height))

            all_polygons_road = all_polygons_road_copy.copy()


            max_track_id = 0
            for track in root.findall('track'):
                if int(track.attrib['id']) > max_track_id:
                    max_track_id = int(track.attrib['id'])

            max_track_id += 1

            # create new track for each road polygon in each frame
            for i in range(len(frames)):
                
                for j in range(len(all_polygons_road[i])): # Process road polylines # j=1
                    for each_road_polygon in range(len(all_polygons_road[i][j])):
                        # print(f"len of all_polygons_road: {len(all_polygons_road)}, len(all_polygons_road[i]: {len(all_polygons_road[i])}, len(all_polygons_road[i][j]): {len(all_polygons_road[i][j])}, each_road_polygon: {each_road_polygon}")
                        points = ''
                        num = 0
                        for l in range(len(all_polygons_road[i][j][each_road_polygon])):
                            num += 1
                            points += str(float(all_polygons_road[i][j][each_road_polygon][l][0])) + ',' + str(float(all_polygons_road[i][j][each_road_polygon][l][1])) + ';'
                        if(num <= 2):
                            continue
                        # assoc_id = 1
                        # create new track
                        new_track = ET.SubElement(root, 'track')
                        new_track.set('id', str(max_track_id))
                        max_track_id += 1
                        new_track.set('label', 'road')
                        new_track.set('source', 'file')

                        # create new tag polygon
                        new_polygon = ET.SubElement(new_track, 'polyline')
                        # points attribute is of form "x1,y1:x2,y2:x3,y3:x4,y4"
                        # remove last semicolon
                        frame_num = i*5
                        new_polygon.set('frame', str(i*5))
                        new_polygon.set('keyframe', '1')
                        new_polygon.set('outside', '0')
                        new_polygon.set('occluded', '0')
                        points = points[:-1]
                        new_polygon.set('points', points.replace('\n', ''))
                        new_polygon.set('z_order', '0')

                        if frame_num == (len(frames) * 5) - 5: # otherwise while uploading the xml, your cvat will throw error that "xx" frame doesn't exist
                            break

                        # creating polygon 2 with outside = 1
                        new_polygon = ET.SubElement(new_track, 'polyline')
                        new_polygon.set('frame', str((i+1)*5))
                        new_polygon.set('keyframe', '1')
                        new_polygon.set('outside', '1')
                        new_polygon.set('occluded', '0')
                        new_polygon.set('points', points.replace('\n', ''))
                        new_polygon.set('z_order', '0')

            # write to xml file
            tree.write(output_xml_folder + os.path.splitext(video_file)[0] + '.xml')
            print('XML file created successfully for', os.path.splitext(video_file)[0] + '.xml')
            print("-"*50)
            print("-"*50)
            print("\n\n\n\n\n\n")

            # Release the video capture object
            cap.release()
            gc.collect()  # Final garbage collection
            # out.release()
            cv2.destroyAllWindows()
