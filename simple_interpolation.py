import numpy as np


def auto_interpolate_bboxes(bboxes, num_frames):
    if not bboxes:
        raise ValueError("At least one bbox is required")

    if len(bboxes) == 1:
        return bboxes * num_frames

    num_keyframes = len(bboxes)
    keyframe_indices = np.linspace(0, num_frames - 1, num_keyframes).astype(int)

    result = []
    for t in range(num_frames):
        if t <= keyframe_indices[0]:
            result.append(bboxes[0])
        elif t >= keyframe_indices[-1]:
            result.append(bboxes[-1])
        else:
            for i in range(len(keyframe_indices) - 1):
                if keyframe_indices[i] <= t <= keyframe_indices[i + 1]:
                    t1 = keyframe_indices[i]
                    t2 = keyframe_indices[i + 1]
                    bbox1 = bboxes[i]
                    bbox2 = bboxes[i + 1]
                    alpha = (t - t1) / (t2 - t1) if t2 > t1 else 0
                    interpolated = [
                        bbox1[j] + (bbox2[j] - bbox1[j]) * alpha
                        for j in range(4)
                    ]
                    result.append(interpolated)
                    break

    return result


def auto_interpolate_trajectory(points, num_frames):
    if not points:
        raise ValueError("At least one point is required")

    if len(points) == 1:
        return points * num_frames

    num_keyframes = len(points)
    keyframe_indices = np.linspace(0, num_frames - 1, num_keyframes).astype(int)

    result = []
    for t in range(num_frames):
        if t <= keyframe_indices[0]:
            result.append(points[0])
        elif t >= keyframe_indices[-1]:
            result.append(points[-1])
        else:
            for i in range(len(keyframe_indices) - 1):
                if keyframe_indices[i] <= t <= keyframe_indices[i + 1]:
                    t1 = keyframe_indices[i]
                    t2 = keyframe_indices[i + 1]
                    point1 = points[i]
                    point2 = points[i + 1]
                    alpha = (t - t1) / (t2 - t1) if t2 > t1 else 0
                    interpolated = [
                        point1[j] + (point2[j] - point1[j]) * alpha
                        for j in range(2)
                    ]
                    result.append(interpolated)
                    break

    return result
