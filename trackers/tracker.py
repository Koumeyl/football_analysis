from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
import json
class Tracker:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
            if self.model:
                print("Model loaded successfully.")
            else:
                # This else block might not be very useful, as the YOLO constructor likely raises an exception if it fails.
                print("Model failed to load. No exception raised, but the model is None.")
        except Exception as e:
            # It's good to catch exceptions to understand if there's a loading error.
            print(f"Failed to load the model due to an exception: {e}")

        device = 'cuda' #if torch.cuda.is_available() else 'cpu'

        model = YOLO(model_path).to(device)
        self.model = model
        self.tracker = sv.ByteTrack()


    
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 100
        detection_threshold = 5  # Minimum detections per batch to continue
        min_detections_per_frame = 1  # Minimum detections per frame
        detections = []
        len_frames = len(frames)

        for i in range(0, len_frames, batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.25)
            frame_detection_counts = [len(frame) for frame in detections_batch]
            
            # Count frames with detections below the per-frame threshold
            low_detection_frames = sum(1 for count in frame_detection_counts if count < min_detections_per_frame)
            

            # # Stop if too many frames have low detections
            # if low_detection_frames > detection_threshold:
            #     print(f"Stopping detection at batch {i//batch_size} due to {low_detection_frames} low detection frames.")
            #     break
            #     continue

            detections.extend(detections_batch)
            
        return detections



    def get_player_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        
        tracks = {
            "players": [{} for _ in frames],
            "referees": [{} for _ in frames],
            "ball": [{} for _ in frames],
            "skipped": []
        }

        for frame_num, detection in enumerate(detections):
            if detection is None or len(detection) == 0:
                tracks["skipped"].append(frame_num)
                continue

            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            try:
                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                if detection_with_tracks is None:
                    print(f"Frame {frame_num}: No valid tracks")
                    tracks["skipped"].append(frame_num)
                    continue

                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]

                    if len(frame_detection) > 4:
                        track_id = frame_detection[4]
                    else:
                        track_id = None

                    if cls_id == cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    if cls_id == cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    for frame_detection in detection_supervision:
                        bbox = frame_detection[0].tolist()
                        cls_id = frame_detection[3]

                        if cls_id == cls_names_inv['ball']:
                            tracks["ball"][frame_num][1] = {"bbox":bbox}
            except Exception as e:
                print(f"An error occurred while processing frame {frame_num}: {e}")
                tracks["skipped"].append(frame_num)

        return tracks



    def get_pitch_tracks(self, frames, read_from_stub=False, stub_path=None):
        with open('detailed_debug_log.txt', 'w') as log_file:
            if read_from_stub and stub_path is not None and os.path.exists(stub_path):
                with open(stub_path, 'rb') as f:
                    tracks = pickle.load(f)
                log_file.write("Loaded tracks from stub.\n")
                return tracks

            detections = self.detect_frames(frames)
            log_file.write(f"Total frames processed: {len(frames)}\n")
            # log_file.write(f"Detections: {detections}\n")

            tracks = {
                "18Yard": [{} for _ in frames],
                "18Yard Circle": [{} for _ in frames],
                "5Yard": [{} for _ in frames],
                "First Half Central Circle": [{} for _ in frames],
                "First Half Field": [{} for _ in frames],
                "Second Half Central Circle": [{} for _ in frames],
                "Second Half Field": [{} for _ in frames],
                "skipped": []
            }

            for frame_num, detection in enumerate(detections):
                # log_file.write(f"Processing frame {frame_num} with detection: {detection}\n")
                if detection is None or len(detection) == 0:
                    tracks["skipped"].append(frame_num)
                    log_file.write(f"Frame {frame_num}: No valid detections, skipping.\n")
                    continue

                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}
                log_file.write(f"Class Names Inverse Mapping: {cls_names_inv}\n")

                detection_supervision = sv.Detections.from_ultralytics(detection)
                log_file.write(f"Detection Supervision for frame {frame_num}: {detection_supervision}\n")

                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
                if detection_supervision is None:
                    tracks["skipped"].append(frame_num)
                    log_file.write(f"Frame {frame_num}: No valid tracks, skipping.\n")
                    continue
                try:    
                    for frame_detection in detection_supervision:
                        bbox = frame_detection[0].tolist()
                        cls_id = frame_detection[3]

                        if len(frame_detection) > 4:
                            track_id = frame_detection[4]
                        else:
                            track_id = None

                        # Using the class IDs provided earlier
                        if cls_id == cls_names_inv['18Yard']:
                            tracks["18Yard"][frame_num][track_id] = {"bbox": bbox}
                        elif cls_id == cls_names_inv['18Yard Circle']:
                            tracks["18Yard Circle"][frame_num][track_id] = {"bbox": bbox}
                        elif cls_id == cls_names_inv['5Yard']:
                            tracks["5Yard"][frame_num][track_id] = {"bbox": bbox}
                        elif cls_id == cls_names_inv['First Half Central Circle']:
                            tracks["First Half Central Circle"][frame_num][track_id] = {"bbox": bbox}
                        elif cls_id == cls_names_inv['First Half Field']:
                            tracks["First Half Field"][frame_num][track_id] = {"bbox": bbox}
                        elif cls_id == cls_names_inv['Second Half Central Circle']:
                            tracks["Second Half Central Circle"][frame_num][track_id] = {"bbox": bbox}
                        elif cls_id == cls_names_inv['Second Half Field']:
                            tracks["Second Half Field"][frame_num][track_id] = {"bbox": bbox}
                except Exception as e:
                    print(f"An error occurred while processing frame {frame_num}: {e}")
                    tracks["skipped"].append(frame_num)
                    log_file.write(f"An error occurred while processing frame {frame_num}: {e}")

            log_file.write("Completed processing all frames.\n")
        return tracks



    def get_key_points_tracks(self, frames, read_from_stub=False, stub_path=None):
        labels = [
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
            "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
            "14", "19"
        ]
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        
        # Initialize the tracks dictionary
        tracks = {
            "keypoints": [{} for _ in frames],
            "skipped": []
        }

        with open('detections_log.txt', 'w') as f:
            for frame_num, detection in enumerate(detections):
                if detection is None or len(detection) == 0:
                    tracks["skipped"].append(frame_num)
                    f.write(f"Frame {frame_num}: No detection\n")
                    continue

                # Convert detection to supervision format
                detection_supervision = sv.KeyPoints.from_ultralytics(detection)
                f.write(f"Frame {frame_num} detection supervision: {detection_supervision}\n")

                if not hasattr(detection_supervision, 'xy'):
                    tracks["skipped"].append(frame_num)
                    f.write(f"Frame {frame_num}: No keypoints attribute\n")
                    continue

                keypoints = detection_supervision.xy  # Access the keypoints
                confidences = detection_supervision.confidence  # Access the confidence scores
                f.write(f"Frame {frame_num} raw keypoints: {keypoints}\n")
                f.write(f"Frame {frame_num} keypoint confidences: {confidences}\n")

                frame_data = {}
                for label, (keypoint, confidence) in zip(labels, zip(keypoints[0], confidences[0])):
                    x, y = keypoint  # Extract x and y coordinates
                    frame_data[label] = {"position": (float(x), float(y)), "confidence": float(confidence)}
                    f.write(f"Frame {frame_num} Keypoint {label}: x={x}, y={y}, confidence={confidence}\n")
                
                tracks["keypoints"][frame_num] = frame_data

        return tracks





    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control=None):
        output_video_frames = []
        num_tracks = len(tracks["players"])  # Assuming the number of frames with track data
        print(tracks["keypoints"])


        for frame_num, frame in enumerate(video_frames):
            if frame_num >= num_tracks:
                output_video_frames.append(frame)
                continue
            frame = frame.copy()

            if "keypoints" in tracks and frame_num < len(tracks["keypoints"]) and tracks["keypoints"][frame_num]:
                keypoint_dict = tracks["keypoints"][frame_num]
                print(f'Frame {frame_num} keypoints:', keypoint_dict)
                for idx, keypoint_info in keypoint_dict.items():
                    x, y = keypoint_info["position"]
                    confidence = keypoint_info["confidence"]
                    if x > 0 and y > 0:  # Ensure the keypoints are within the frame boundaries
                        # Draw a circle at each keypoint location
                        cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)  # Green color
                        # Optionally, display the index and confidence if needed
                        cv2.putText(frame, f'{idx}', (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 1)
                        if frame_num % 50 == 0:
                            cv2.imwrite(f'output_frames_key_points/frame_{frame_num:04d}.png', frame)


            # Existing code to draw other annotations like players, referees, etc.
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # # Assuming there's a method to draw rectangles or other shapes for pitch elements
            # for pitch_element_name in ["18Yard", "18Yard Circle", "5Yard", "First Half Central Circle", "First Half Field", "Second Half Central Circle", "Second Half Field"]:
            #     if pitch_element_name in tracks:
            #         pitch_dict = tracks[pitch_element_name][frame_num]
            #         for track_id, pitch in pitch_dict.items():
            #             x, y, w, h = pitch['bbox']
            #             color = (255, 255, 0)  # Yellow color in BGR format
            #             cv2.circle(frame, (int(x), int(y)), 3, color, -1)  # Example for drawing pitch elements
            

            output_video_frames.append(frame)

        return output_video_frames

