
#!/usr/bin/env python3
import sys, time, os
import cv2
import numpy as np
import joblib
from scipy import fftpack

# load model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", "scaler.joblib")
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def global_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def patch_laplacian_stats(gray, patch=64):
    h,w = gray.shape
    vals = []
    for y in range(0, h, patch):
        for x in range(0, w, patch):
            p = gray[y:y+patch, x:x+patch]
            if p.size == 0: continue
            vals.append(float(cv2.Laplacian(p, cv2.CV_64F).var()))
    if len(vals)==0:
        return 0.0,0.0,0.0
    arr = np.array(vals)
    return float(np.percentile(arr,10)), float(np.percentile(arr,50)), float(np.percentile(arr,90))

def fft_mid_high_ratio(gray):
    sh = gray.shape
    if max(sh) > 512:
        scale = 512.0 / max(sh)
        gray = cv2.resize(gray, (int(sh[1]*scale), int(sh[0]*scale)), interpolation=cv2.INTER_AREA)
    f = fftpack.fft2(gray)
    fshift = fftpack.fftshift(f)
    mag = np.abs(fshift)
    total = mag.sum() + 1e-9
    cy, cx = mag.shape[0]//2, mag.shape[1]//2
    r = min(mag.shape)//8
    yy,xx = np.ogrid[:mag.shape[0], :mag.shape[1]]
    dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    low_mask = dist <= r
    mid_mask = (dist > r) & (dist <= r*3)
    high_mask = dist > r*3
    mid = mag[mid_mask].sum()
    high = mag[high_mask].sum()
    return float((mid+high)/total), float(high/(mid+1e-9))

def saturation_variance(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1].astype(np.float32)
    return float(np.var(sat))

def edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.sum() / (255.0 * gray.size))

def specular_count(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    _,th = cv2.threshold(v, 240, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if 5 < area < 2000:
            count += 1
    return float(count)

def extract_features(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to read " + path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    global_lap = global_laplacian(gray)
    p10,p50,p90 = patch_laplacian_stats(gray, patch=64)
    midhigh, high_ratio = fft_mid_high_ratio(gray)
    sat_var = saturation_variance(bgr)
    ed = edge_density(gray)
    spec = specular_count(bgr)
    feats = [global_lap, p10, p50, p90, midhigh, high_ratio, sat_var, ed, spec]
    return np.array(feats, dtype=np.float32).reshape(1, -1)

def predict(path, verbose=True):
    t0 = time.time()
    feats = extract_features(path)
    feats_scaled = scaler.transform(feats)
    prob = float(clf.predict_proba(feats_scaled)[0,1])
    latency_ms = (time.time() - t0) * 1000.0
    if verbose:
        print(f"DEBUG: image={path} prob_screen={prob:.4f} latency_ms={latency_ms:.1f}")
    print(f"{prob:.4f}")
    return prob

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py some_image.jpg")
        sys.exit(1)
    img_path = sys.argv[1]
    try:
        predict(img_path, verbose=True)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(2)
