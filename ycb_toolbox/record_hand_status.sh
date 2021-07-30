sleep 5
rosbag record --duration=20 \
	/tf /tf_static \
	/k4a/tracked_body_data \
	/k4a/hand_status \
	/k4a/left_hand_marker_array \
	/k4a/left_hand_status \
	/k4a/right_hand_marker_array \
	/k4a/right_hand_status \
	/k4a/tracked_left_hand_data \
	/k4a/tracked_right_hand_data \
	/k4a/rgb/camera_info \
	/k4a/rgb/image_raw \
	/k4a/depth_to_rgb/image_raw \
	/k4a/body_tracking_data \
	/poserbpf/00/info \
	/poserbpf/01/info \
