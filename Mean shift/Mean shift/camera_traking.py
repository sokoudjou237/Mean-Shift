import pygame
import json
import numpy as np
import pygame.camera
import mean_shift


def load_config():
    try:
        with open("config.json", 'r') as js_file:
            js_config = json.load(js_file)
        size = js_config["SIZE"]
        fps = js_config["FPS"]
    except Exception as e:
        with open("config.json", 'w') as js_file:
            js_config = {
                "SIZE": (640, 480),
                "FPS": 60,
            }
            js_string = json.dumps(js_config, sort_keys=True,
                                   indent=4, separators=(',', ': '))
            js_file.write(js_string)
        size = js_config["SIZE"]
        fps = js_config["FPS"]
    return size, fps

pygame.init()
pygame.font.init()

FONT = pygame.font.SysFont(name="arial", size=16)
CLOCK = pygame.time.Clock()
SIZE, FPS = load_config()
DISPLAY = pygame.display.set_mode(SIZE)
CAM = None
RECT = None
RECT_DRAWING = False
TRACKER: mean_shift.Mean_Shift_Tracker = None

try:
    pygame.camera.init()
    _SYS_CAMS = pygame.camera.list_cameras()
    CAM = pygame.camera.Camera(_SYS_CAMS[1], SIZE)
    CAM.start()
except BaseException as e:
    raise BaseException("Pas de caméra disponible !" + str(e))

def capture():
    """Capture une image"""
    if CAM is not None:
        frame = CAM.get_image()
    return frame

def surf_to_ndarray(surf: pygame.Surface):
    """Convertit une image pygame en format numpy (pour cv2)"""
    frame_np = np.array(pygame.surfarray.pixels3d(surf))
    frame_np = np.transpose(frame_np, (1, 0, 2))
    return frame_np

def init_meanshift(cx, cy, w, h):
    """Initialise le détecteur mean_shift"""
    frame = CAM.get_image()
    frame_np = surf_to_ndarray(frame)
    tracker = mean_shift.Mean_Shift_Tracker(cx, cy, w, h)
    tracker.update_target_model(frame_np)
    return tracker

def render(frame: pygame.Surface):
    """Rendu de l'écran"""
    global DISPLAY, SIZE
    if CAM is not None:
        frame = CAM.get_image()
        if frame.get_size() != SIZE:
            SIZE = frame.get_size()
            DISPLAY = pygame.display.set_mode(SIZE)
        DISPLAY.blit(frame, (0, 0))
        if RECT is not None:
            pygame.draw.rect(DISPLAY, "red", RECT, 10)
        CLOCK.tick(FPS)
        pygame.display.flip()
        pygame.display.set_caption(f"FPS: {CLOCK.get_fps():.3}")

def track(frame: np.ndarray):
    """Suivi de la zone cible"""
    global RECT
    if TRACKER is not None and not RECT_DRAWING:
        try:
            TRACKER.perform_mean_shift(frame)
            x1 = TRACKER.curr_cx - (TRACKER.curr_width // 2)
            y1 = TRACKER.curr_cy - (TRACKER.curr_height // 2)
            RECT = [x1, y1, TRACKER.curr_width, TRACKER.curr_height]
        except Exception as e:
            print(f"Erreur de suivi : {e}")

def event_handler():
    global RECT, RECT_DRAWING, TRACKER
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            RECT_DRAWING = True
            RECT = [event.pos[0], event.pos[1], 0, 0]
        if event.type == pygame.MOUSEMOTION:
            if RECT_DRAWING:
                w = RECT[0] - event.pos[0]
                h = RECT[1] - event.pos[1]
                RECT[0] = event.pos[0] if w > 0 else RECT[0]
                RECT[1] = event.pos[1] if h > 0 else RECT[1]
                RECT = [RECT[0], RECT[1], abs(w), abs(h)]
        if event.type == pygame.MOUSEBUTTONUP:
            RECT_DRAWING = False
            _, _, w, h = RECT
            if w % 2 == 0:
                w += 1
            if h % 2 == 0:
                h += 1
            cx = RECT[0] + (w // 2)
            cy = RECT[1] + (h // 2)
            # Assurer que la hauteur et la largeur de la cible sont impaires pour faciliter le traitement
            print(cx, cy, w, h)
            TRACKER = init_meanshift(cx, cy, w, h)

def mainloop():
    while True:
        frame = capture()
        render(frame)
        track(surf_to_ndarray(frame))
        event_handler()

if __name__ == "__main__":
    try:
        mainloop()
    finally:
        # Enregistrement des données après l'arrêt de la boucle principale
        import json

        # Enregistrement des positions de suivi
        with open('tracking_positions.json', 'w') as f:
            json.dump(mean_shift.tracking_positions, f)

        # Enregistrement des similarités
        with open('tracking_similarities.json', 'w') as f:
            json.dump(mean_shift.tracking_similarities, f)
