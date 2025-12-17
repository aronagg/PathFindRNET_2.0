import cv2
from typing import Iterable, Sequence, Tuple, Optional

# annos: iterable of (x1,y1,x2,y2, cls, conf, id) - id can be None
def draw_annotations(
    img,
    annos: Iterable[Tuple[float, float, float, float, int, float, Optional[int]]],
    colors: Sequence[Optional[Tuple[int, int, int]]],
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """Draw boxes + labels on image in-place.

    colors is a sequence indexed by class_id; entries may be None for unused ids.
    """
    for x1, y1, x2, y2, cls, conf, tid in annos:
        if cls is not None and 0 <= int(cls) < len(colors) and colors[int(cls)]:
            col = tuple(map(int, colors[int(cls)]))
        else:
            col = (200, 200, 200)
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, col, 2)
        label_cls = None
        if class_names and cls is not None and 0 <= int(cls) < len(class_names):
            label_cls = class_names[int(cls)]
        label_text = label_cls if label_cls is not None else str(cls)
        if tid is None or tid < 0:
            txt = f"{label_text} {conf:.2f}"
        else:
            txt = f"{label_text} {conf:.2f} ID:{tid}"
        cv2.putText(img, txt, (p1[0], max(15, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def show_frame(window_name: str, img, wait: int = 1) -> bool:
    """Show image and return True if user requested quit (pressed 'q')."""
    cv2.imshow(window_name, img)
    key = cv2.waitKey(wait) & 0xFF
    return key == ord("q")