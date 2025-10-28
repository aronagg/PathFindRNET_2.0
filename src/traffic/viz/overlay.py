import cv2


def draw_overlay(frame, pred_exit_point=None, alt_exit_point=None):
    if pred_exit_point is not None:
        cv2.circle(frame, tuple(map(int, pred_exit_point)), 5, (0, 255, 0), -1)  # green
    if alt_exit_point is not None:
        cv2.circle(frame, tuple(map(int, alt_exit_point)), 5, (0, 0, 255), -1)  # red
    return frame
