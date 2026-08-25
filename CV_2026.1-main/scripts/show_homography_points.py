import json
import os
import cv2


HOMOGRAPHY_JSON = "dst_homo_2 bkp.json"
REFRESH_INTERVAL_MS = 50
MARGIN_PX = 40

def load_points(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = {}
    for p in data.get("dst_homo", []):
        try:
            pid = int(p["id"])
            x = float(p["x"]) 
            y = float(p["y"]) 
        except Exception:
            continue
        pts[pid] = (x, y)
    return pts


def draw_points(img, points, circle_color=(0, 255, 0), text_color=(0, 0, 255), offset=(0, 0)):
    off_x, off_y = offset
    for pid, (x, y) in points.items():
        cx, cy = int(round(x + off_x)), int(round(y + off_y))
        cv2.circle(img, (cx, cy), 6, circle_color, -1)
        cv2.putText(img, str(pid), (cx + 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)


def show_image_with_points(image_path: str, points_path: str, window_name: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_mtime = None
    while True:
        try:
            mtime = os.path.getmtime(points_path)
        except FileNotFoundError:
            mtime = None

        if mtime != last_mtime:
            last_mtime = mtime
            points = load_points(points_path) if mtime is not None else {}
            h, w = img.shape[:2]
            img_copy = cv2.copyMakeBorder(
                img,
                MARGIN_PX,
                MARGIN_PX,
                MARGIN_PX,
                MARGIN_PX,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            draw_points(img_copy, points, offset=(MARGIN_PX, MARGIN_PX))
            cv2.imshow(window_name, img_copy)

        key = cv2.waitKey(REFRESH_INTERVAL_MS)
        if key != -1:
            break


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    image_path = os.path.join(root, "assets", "Background_2.png")
    points_path = os.path.join(root, "assets", HOMOGRAPHY_JSON)

    print("Mostrando a janela — atualizando em tempo real. Pressione qualquer tecla para fechar...")
    show_image_with_points(image_path, points_path, os.path.basename(points_path))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()