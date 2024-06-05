from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import json
import pandas as pd



def main():
    # Read Video
    video_frames = read_video('input_videos/antwerp_angle.mp4')

    #Initialize Trackers for each model
    player_tracker = Tracker('models/best-4000.pt')
    pitch_part_tracker = Tracker('models/best-4000-pitch-part-detection.pt')
    key_points_tracker = Tracker('models/Field_Key_Points.pt')
    #Get object tracks from both models
    player_tracks = player_tracker.get_player_tracks(video_frames)
    pitch_part_tracks = pitch_part_tracker.get_pitch_tracks(video_frames)
    key_points_tracks = key_points_tracker.get_key_points_tracks(video_frames)

    print(player_tracks.keys())
    print(pitch_part_tracks.keys())
    print(key_points_tracks.keys())






    # for track_id, player in tracks['players'][2].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     cv2.imwrite(f'output_videos/image.jpg', cropped_image)
    #     break



    # # Get object positions 
    # player_tracker.add_position_to_tracks(player_tracks)

    
    # #camera movement estimator
    # camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    # camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
    #                                                                             read_from_stub=False,
    #                                                                             stub_path=None)
    # camera_movement_estimator.add_adjust_positions_to_tracks(player_tracks,camera_movement_per_frame)


    # # # # View Trasnformer
    # view_transformer = ViewTransformer()
    # view_transformer.add_transformed_position_to_tracks(player_tracks)

    # Interpolate Ball Positions
    player_tracks["ball"] = player_tracker.interpolate_ball_positions(player_tracks["ball"])

    # # Speed and distance estimator
    # speed_and_distance_estimator = SpeedAndDistance_Estimator()
    # speed_and_distance_estimator.add_speed_and_distance_to_tracks(player_tracks)


    
    team_assigner = TeamAssigner()
    first_frame_with_players = next((i for i, p in enumerate(player_tracks['players']) if p), None)

    if first_frame_with_players is not None:
        team_assigner.assign_team_color(video_frames[first_frame_with_players], player_tracks['players'][first_frame_with_players])
    else:
        print("No players detected in any frame.")

    for frame_num, player_track in enumerate(player_tracks['players']):
        if frame_num < len(video_frames):  # Ensure the frame number is within the range of available video frames
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
                player_tracks['players'][frame_num][player_id]['team'] = team 
                player_tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        else:
            print(f"Frame number {frame_num} is out of range of the available video frames.")

    
    #Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(player_tracks['players']):
        ball_bbox = player_tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            player_tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(player_tracks['players'][frame_num][assigned_player]['team'])
        else:
            continue
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)
    print(key_points_tracks)

    # Draw output 
    ## Draw object Tracks
    ## Draw Camera movement
    player_tracks.update(pitch_part_tracks)
    player_tracks.update(key_points_tracks)
    # output_video_frames = player_tracker.draw_annotations(video_frames, player_tracks,team_ball_control)
    output_video_frames = player_tracker.draw_annotations(video_frames, player_tracks)

    all_data = []
    skipped_frames = set(player_tracks.get("skipped", []))  # Get skipped frame numbers

    # Collect data for all entities in each frame
    for frame_num in range(len(video_frames)):
        if frame_num in skipped_frames:
            # Record that this frame was skipped due to low or no detections
            all_data.append({
                'Frame': frame_num,
                'ID': 'None',
                'Type': 'Skipped',
                'Info': 'No valid detections'
            })
            continue

        frame_players = player_tracks['players'][frame_num] if frame_num < len(player_tracks['players']) else {}
        frame_balls = player_tracks['ball'][frame_num] if 'ball' in player_tracks and frame_num < len(player_tracks['ball']) else {}
        frame_referees = player_tracks['referees'][frame_num] if 'referees' in player_tracks and frame_num < len(player_tracks['referees']) else {}

        # Add pitch parts explicitly
        frame_pitch_18Yard = pitch_part_tracks['18Yard'][frame_num] if frame_num < len(pitch_part_tracks['18Yard']) else {}
        frame_pitch_18Yard_Circle = pitch_part_tracks['18Yard Circle'][frame_num] if frame_num < len(pitch_part_tracks['18Yard Circle']) else {}
        frame_pitch_5Yard = pitch_part_tracks['5Yard'][frame_num] if frame_num < len(pitch_part_tracks['5Yard']) else {}
        frame_pitch_First_Half_Central_Circle = pitch_part_tracks['First Half Central Circle'][frame_num] if frame_num < len(pitch_part_tracks['First Half Central Circle']) else {}
        frame_pitch_First_Half_Field = pitch_part_tracks['First Half Field'][frame_num] if frame_num < len(pitch_part_tracks['First Half Field']) else {}
        frame_pitch_Second_Half_Central_Circle = pitch_part_tracks['Second Half Central Circle'][frame_num] if frame_num < len(pitch_part_tracks['Second Half Central Circle']) else {}
        frame_pitch_Second_Half_Field = pitch_part_tracks['Second Half Field'][frame_num] if frame_num < len(pitch_part_tracks['Second Half Field']) else {}

        #Add keypoints 
        frame_keypoints = key_points_tracks['keypoints'][frame_num] if frame_num < len(key_points_tracks['keypoints']) else {}

        if not frame_players and not frame_balls and not frame_referees and not frame_pitch_18Yard and not frame_pitch_18Yard_Circle and not frame_pitch_5Yard and not frame_pitch_First_Half_Central_Circle and not frame_pitch_First_Half_Field and not frame_pitch_Second_Half_Central_Circle and not frame_pitch_Second_Half_Field:
            # Record that this frame had no detections
            all_data.append({
                'Frame': frame_num,
                'ID': 'None',
                'Type': 'Empty',
                'Info': 'No detections'
            })
        else:
            # Collect data from players, balls, referees, and pitch parts
            for player_id, player_info in frame_players.items():
                team = team_assigner.get_player_team(video_frames[frame_num], player_info['bbox'], player_id)
                team_color = team_assigner.team_colors.get(team, (0, 0, 0))  # Default color if not found

                player_data = {
                    'Frame': frame_num,
                    'ID': player_id,
                    'Type': 'Player',
                    'Has Ball': player_info.get('has_ball', False),
                    'Team': team,
                    'Team Color': team_color,
                    'BBox': player_info['bbox']
                }
                all_data.append(player_data)
        
            for ball_id, ball_info in frame_balls.items():
                ball_data = {
                    'Frame': frame_num,
                    'ID': ball_id,
                    'Type': 'Ball',
                    'BBox': ball_info['bbox']
                }
                all_data.append(ball_data)

            for referee_id, referee_info in frame_referees.items():
                referee_data = {
                    'Frame': frame_num,
                    'ID': referee_id,
                    'Type': 'Referee',
                    'BBox': referee_info['bbox']
                }
                all_data.append(referee_data)

            # Collect pitch part data
            for part_name, part_info in frame_pitch_18Yard.items():
                part_data = {
                    'Frame': frame_num,
                    'ID': '18Yard',
                    'Type': 'Pitch Part',
                    'BBox': part_info['bbox']
                }
                all_data.append(part_data)
            
            for part_name, part_info in frame_pitch_18Yard_Circle.items():
                part_data = {
                    'Frame': frame_num,
                    'ID': '18Yard Circle',
                    'Type': 'Pitch Part',
                    'BBox': part_info['bbox']
                }
                all_data.append(part_data)
            
            for part_name, part_info in frame_pitch_5Yard.items():
                part_data = {
                    'Frame': frame_num,
                    'ID': '5Yard',
                    'Type': 'Pitch Part',
                    'BBox': part_info['bbox']
                }
                all_data.append(part_data)
            
            for part_name, part_info in frame_pitch_First_Half_Central_Circle.items():
                part_data = {
                    'Frame': frame_num,
                    'ID': 'First Half Central Circle',
                    'Type': 'Pitch Part',
                    'BBox': part_info['bbox']
                }
                all_data.append(part_data)
            
            for part_name, part_info in frame_pitch_First_Half_Field.items():
                part_data = {
                    'Frame': frame_num,
                    'ID': 'First Half Field',
                    'Type': 'Pitch Part',
                    'BBox': part_info['bbox']
                }
                all_data.append(part_data)
            
            for part_name, part_info in frame_pitch_Second_Half_Central_Circle.items():
                part_data = {
                    'Frame': frame_num,
                    'ID': 'Second Half Central Circle',
                    'Type': 'Pitch Part',
                    'BBox': part_info['bbox']
                }
                all_data.append(part_data)
            
            for part_name, part_info in frame_pitch_Second_Half_Field.items():
                part_data = {
                    'Frame': frame_num,
                    'ID': 'Second Half Field',
                    'Type': 'Pitch Part',
                    'BBox': part_info['bbox']
                }
                all_data.append(part_data)

        for keypoint_id, keypoint_info in frame_keypoints.items():
            keypoint_data = {
                'Frame': frame_num,
                'ID': keypoint_id,
                'Type': 'KeyPoint',
                'Position': keypoint_info['position'],
                'Confidence': keypoint_info['confidence']
            }
            all_data.append(keypoint_data)

    # Convert the list of all data into a DataFrame
    df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file
    df.to_csv('all_tracking_data.csv', index=False)
    print("All tracking data saved to all_tracking_data.csv")



    # output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    # speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,player_tracks)
    print(player_tracks.keys())
    print(player_tracks.items())
    print(pitch_part_tracks.keys())
    print(pitch_part_tracks)



    import os

    output_dir = 'output_videos'
    output_file = os.path.join(output_dir, 'output_video.avi')

    # Create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the video
    save_video(output_video_frames, output_file)

    # Open the video file
    os.startfile(output_file)


if __name__ == '__main__':
    main()


# def main():
#     # Read Video
#     video_frames = read_video('input_videos/antwerp_angle.mp4')

#     # Initialize Trackers for each model
#     player_tracker = Tracker('models/best-4000.pt')
#     pitch_part_tracker = Tracker('models/best-4000-pitch-part-detection.pt')
#     # Get object tracks from both models
#     player_tracks = player_tracker.get_player_tracks(video_frames)
#     pitch_part_tracks = pitch_part_tracker.get_pitch_tracks(video_frames)
#     print(player_tracks.keys())
#     print(pitch_part_tracks.keys())


#     player_tracks.update(pitch_part_tracks)


#     # Draw annotations for both player and pitch parts
#     # Ensure draw_annotations is capable of drawing the new pitch part classes
#     output_video_frames = player_tracker.draw_annotations(video_frames, player_tracks)

#     # # Save video
#     save_video(output_video_frames, 'output_videos/output_video.avi')

# if __name__ == '__main__':
#     main()
