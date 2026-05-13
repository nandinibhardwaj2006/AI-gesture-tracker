import cv2
import mediapipe as mp
import numpy as np
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
m_l= 61
m_r= 291
m_t= 13
m_b = 14
left_b_i=107
left_b_o= 70
right_b_i= 336
right_b_o= 300
left_e_t=159
left_e_b=145
right_e_t=386
right_e_b=374
n= 1
c= 152
f=10
def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])
def compute_confidence_score(landmarks, w, h):
    scores = {}
    nose     = get_point(landmarks, n, w, h)
    chin     = get_point(landmarks, c, w, h)
    forehead = get_point(landmarks, f, w, h)

    upper = nose[1] - forehead[1]  
    lower = chin[1] - nose[1]       
    ratio = upper / (lower + 1e-6)
    scores["head_pitch"] = float(np.clip((ratio - 0.4) / 0.7, 0, 1))
    ml = get_point(landmarks, m_l, w, h)   
    mr = get_point(landmarks, m_r, w, h)  
    mt = get_point(landmarks, m_t, w, h)   
    corner_avg_y = (ml[1] + mr[1]) / 2   
    center_y     = mt[1]                   
    mouth_width  = np.linalg.norm(mr - ml) + 1e-6  
    curve = (center_y - corner_avg_y) / mouth_width
    scores["mouth_curve"] = float(np.clip(0.5 + curve * 3, 0, 1))
    score = float(np.clip(0.5 + curve * 3, 0, 1))
    l_brow_i = get_point(landmarks, left_b_i,  w, h)
    l_brow_o = get_point(landmarks, left_b_o,  w, h)
    r_brow_i = get_point(landmarks, right_b_i, w, h)
    r_brow_o = get_point(landmarks, right_b_o, w, h)
    l_eye_t  = get_point(landmarks, left_e_t,     w, h)
    r_eye_t  = get_point(landmarks, right_e_t,    w, h)
    left_brow_height  = (l_eye_t[1] - ((l_brow_i[1] + l_brow_o[1]) / 2)) / (h + 1e-6)
    right_brow_height = (r_eye_t[1] - ((r_brow_i[1] + r_brow_o[1]) / 2)) / (h + 1e-6)
    avg_brow = (left_brow_height + right_brow_height) / 2
    scores["brow_height"] = float(np.clip(avg_brow * 30, 0, 1))
    le_t = get_point(landmarks, left_e_t,     w, h)
    le_b = get_point(landmarks, left_e_b,  w, h)
    re_t = get_point(landmarks, right_e_t,    w, h)
    re_b = get_point(landmarks, right_e_b, w, h)
    left_open  = abs(le_b[1] - le_t[1]) / (h + 1e-6)
    right_open = abs(re_b[1] - re_t[1]) / (h + 1e-6)
    avg_eye = (left_open + right_open) / 2
    scores["eye_openness"] = float(np.clip(avg_eye * 50, 0, 1))
    weights = {
        "head_pitch":   0.15,
        "mouth_curve":  0.35,
        "brow_height":  0.15,
        "eye_openness": 0.35,
    }
    final = sum(scores[k] * weights[k] for k in weights)
    return final, scores
HISTORY_LEN = 20
score_history = []

def smooth_score(new_score):
    score_history.append(new_score)
    if len(score_history) > HISTORY_LEN:
        score_history.pop(0)
    return sum(score_history) / len(score_history)
def draw_confidence_overlay(frame, score, sub_scores, threshold=0.52):
    h, w = frame.shape[:2]
    label    = "CONFIDENT" if score >= threshold else "UNDER-CONFIDENT"
    color    = (0, 220, 80)  if score >= threshold else (0, 80, 220)
    cv2.putText(frame, f"{label}  ({score:.2f})",
                (12, 42), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)


    x, y = 1, 50
    w, h = 160, 12
    for i, (name, val) in enumerate(sub_scores.items()):
        y = y + 30
        cv2.putText(frame, f"{name}: {val:.2f}",
                    (x, y ),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0 ,0), 4)
def main():
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm = face_landmarks.landmark
                    mp_drawing.draw_landmarks(
                        frame, face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(80, 110, 10), thickness=1, circle_radius=1),
                    )
                    raw_score, sub_scores = compute_confidence_score(lm, w, h)
                    score = smooth_score(raw_score)
                    draw_confidence_overlay(frame, score, sub_scores)
            cv2.imshow("Confidence Detector", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()