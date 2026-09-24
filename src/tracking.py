from grid_tracker import GridTracker
import numpy as np
import cv2
import time
import marker_dectection
import sys
import setting

calibrate = False

if len(sys.argv) > 1:
    if sys.argv[1] == 'calibrate':
        calibrate = True

gelsight_version = 'Bnz'
# gelsight_version = 'HSR'

# cap = cv2.VideoCapture("data/GelSight_Twist_Test.mov")
# cap = cv2.VideoCapture("data/GelSight_Shear_Test.mov")
cap = cv2.VideoCapture(1)


# Resize scale for faster image processing
setting.init()
RESCALE = setting.RESCALE

# Initialize from a complete untouched grid in processed-frame coordinates.
m = GridTracker(setting.N_, setting.M_)

# save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')

if gelsight_version == 'HSR':
    out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (215,215))
else:
    out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (1280//RESCALE,720//RESCALE))

# for i in range(30): ret, frame = cap.read()

def measure_grid(mc, rows=7, cols=9):
    points = np.asarray(mc, dtype=np.float64)

    if len(points) != rows * cols:
        print(f"Need {rows * cols} markers; detected {len(points)}")
        return None

    # For a roughly horizontal grid: group by row, then sort left to right.
    points = points[np.argsort(points[:, 1])]
    grid = points.reshape(rows, cols, 2)
    grid = np.stack([row[np.argsort(row[:, 0])] for row in grid])

    x0, y0 = grid[0, 0]
    dx = np.diff(grid[:, :, 0], axis=1).mean()
    dy = np.diff(grid[:, :, 1], axis=0).mean()

    print("\n# Paste into setting.py (already in processed-frame pixels):")
    print(f"N_ = {rows}")
    print(f"M_ = {cols}")
    print(f"x0_ = {x0:.3f}")
    print(f"y0_ = {y0:.3f}")
    print(f"dx_ = {dx:.3f}")
    print(f"dy_ = {dy:.3f}\n")
    sys.exit()

    return grid

grid_measured = False

while(True):

    # capture frame-by-frame
    ret, frame = cap.read()
    if not(ret):
        break

    frame_raw = frame.copy()

    # resize (or unwarp)
    if gelsight_version == 'HSR':
        frame = marker_dectection.init_HSR(frame)
    else:
        frame = marker_dectection.init(frame)
    # frame = marker_dectection.init_HSR(frame)

    # find marker masks
    mask = marker_dectection.find_marker(frame)

    # find marker centers
    mc = marker_dectection.marker_center(mask, frame)

    if calibrate and not grid_measured:
        grid = measure_grid(mc)
        grid_measured = grid is not None


    if calibrate == False:
        flow = m.update(mc)
        if flow is None:
            status = f"Waiting for {setting.N_ * setting.M_} markers; detected {len(mc)}"
        else:
            Ox, Oy, Cx, Cy, occupied = flow
            valid = occupied >= 0
            # Never draw stale positions for missing or ambiguous detections.
            for i, j in zip(*np.where(valid)):
                start = (int(round(Ox[i, j])), int(round(Oy[i, j])))
                # end = (int(round(Cx[i, j])), int(round(Cy[i, j])))
                arrow_scale = 10
                end = (
                    int(round(Ox[i, j] + arrow_scale * (Cx[i, j] - Ox[i, j]))),
                    int(round(Oy[i, j] + arrow_scale * (Cy[i, j] - Oy[i, j]))),
                )
                # cv2.arrowedLine(frame, start, end, (0, 0, 255), 1, tipLength=0.2)
                cv2.arrowedLine(frame, start, end, (0, 0, 255), 2, tipLength=0.2)
            status = f"Matched {valid.sum()}/{valid.size} | r: reset untouched reference"
        cv2.putText(frame, status, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1)

    mask_img = mask.astype(frame[0].dtype)
    mask_img = cv2.merge((mask_img, mask_img, mask_img))

    # cv2.imshow('raw',frame_raw)
    cv2.imshow('frame',frame)

    if calibrate:
        # Display the mask 
        cv2.imshow('mask',mask_img)

    out.write(frame)

    print(frame.shape)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        m.reset()

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()