import cv2
import numpy as np

# 座席エリアの4点座標（必要に応じて調整してください）
SEAT_AREAS = {
    'left':   [(372, 1212), (734, 1616), (1337, 1203), (842, 977)],
    'center': [(765, 1692), (1565, 1089), (2258, 1389), (1560, 1964)],
    'right':  [(1645, 2057), (2628, 1027), (3335, 977), (3351, 2065)]
}

SEAT_COLORS = {
    'left': (255, 0, 0),
    'center': (0, 255, 0),
    'right': (0, 0, 255)
}

def is_point_in_polygon(point, polygon):
    """
    指定した点が多角形の内側にあるか判定
    Args:
        point (tuple): (x, y)
        polygon (list): [(x1, y1), (x2, y2), ...]
    Returns:
        bool: 内側ならTrue
    """
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0


def assign_seat_to_bbox(bbox, seat_areas=SEAT_AREAS):
    """
    バウンディングボックスの足元がどの座席エリアにあるか判定
    Args:
        bbox (list): [x1, y1, x2, y2]
        seat_areas (dict): 座席エリア定義
    Returns:
        seat_name (str or None): 割り当てられた座席名（なければNone）
        foot (tuple): 足元座標
    """
    x1, y1, x2, y2 = bbox
    foot = (int((x1 + x2) / 2), int(y2))
    for seat, poly in seat_areas.items():
        if is_point_in_polygon(foot, poly):
            return seat, foot
    return None, foot 