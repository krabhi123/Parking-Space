from flask import Flask, request, jsonify
from util import get_parking_spots_bboxes, empty_or_not
import cv2
import numpy as np
import pickle

app = Flask(__name__)

EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("model.p", "rb"))


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


def process_frame(frame, spots, diffs, previous_frame, spots_status):
    for spot_indx, spot in enumerate(spots):
        x1, y1, w, h = spot

        spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

        diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

    if previous_frame is None:
        arr_ = range(len(spots))
    else:
        arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

    for spot_indx in arr_:
        spot = spots[spot_indx]
        x1, y1, w, h = spot

        spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

        spot_status = empty_or_not(spot_crop)

        spots_status[spot_indx] = spot_status

    return spots_status, diffs, frame


@app.route('/parking', methods=['POST'])
def detect_parking_spots():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    npimg = np.fromstring(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    mask = cv2.imread('./mask_1920_1080.png', 0)
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)

    spots_status = [None for j in spots]
    diffs = [None for j in spots]

    previous_frame = None

    spots_status, diffs, frame = process_frame(frame, spots, diffs, previous_frame, spots_status)

    previous_frame = frame.copy()

    response = {
        'spots': []
    }

    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            response['spots'].append({'id': spot_indx, 'status': 'empty', 'x1': x1, 'y1': y1, 'w': w, 'h': h})
        else:
            response['spots'].append({'id': spot_indx, 'status': 'occupied', 'x1': x1, 'y1': y1, 'w': w, 'h': h})

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
