from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import torch

from FastSAM.fastsam import FastSAM, FastSAMPrompt

model = FastSAM('FastSAM.pt')
DEVICE = 'cpu'

#ngambil foto dari kamera
def capture_from_two_cameras(output_folder='images'):
    # Membuat folder jika belum ada
    if not cv2.os.path.exists(output_folder):
        cv2.os.makedirs(output_folder)

    # Mengakses kamera pertama
    cap1 = cv2.VideoCapture(0)  # 0 menunjukkan penggunaan kamera utama, bisa diganti jika memiliki kamera lain

    # Mengakses kamera kedua (misalnya, menggunakan indeks 1 untuk kamera sekunder)
    cap2 = cv2.VideoCapture(0)  # Ganti dengan indeks yang sesuai jika perlu

    # Mengecek apakah kedua kamera dapat diakses
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Satu atau kedua kamera tidak dapat diakses.")
        return

    # Membaca frame dari kamera pertama
    ret1, frame1 = cap1.read()

    # Membaca frame dari kamera kedua
    ret2, frame2 = cap2.read()

    # Menyimpan gambar dari kamera pertama
    if ret1:
        image_path1 = cv2.os.path.join(output_folder, 'bayi-up.jpg')
        cv2.imwrite(image_path1, frame1)
        print(f"Gambar dari kamera pertama telah disimpan di: {image_path1}")
    else:
        print("Error: Gagal membaca frame dari kamera pertama.")

    # Menyimpan gambar dari kamera kedua
    if ret2:
        image_path2 = cv2.os.path.join(output_folder, 'baby_side.jpg')
        cv2.imwrite(image_path2, frame2)
        print(f"Gambar dari kamera kedua telah disimpan di: {image_path2}")
    else:
        print("Error: Gagal membaca frame dari kamera kedua.")

    # Menutup kedua kamera
    cap1.release()
    cap2.release()

#capture_from_two_cameras()

#fungsi perhitungan
def calculate_distance(x1, y1, x2, y2):
  return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def perhitungan_elips(sb_mayor, sb_minor):
  #hitung keliling elips
  keliling_elips=0.5 * math.pi *(sb_mayor + sb_minor)
  return keliling_elips

def calculate_bbox_from_mask(mask):
    indices = np.where(mask > 0)
    y_min = np.min(indices[0])
    y_max = np.max(indices[0])
    x_min = np.min(indices[1])
    x_max = np.max(indices[1])
    y = y_max - y_min
    x = x_max - x_min
    if y > x:
        width = y
    else:
        width = x
    return [x_min, y_min, x_max, y_max], width

#kelas masking fastsam
class FastSam():
  def __init__(self, IMAGE_PATH):
    self.path = IMAGE_PATH
    self.image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)

#===============================================================================
  def get_input_point(self):
    base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(self.path)

    detection_result = detector.detect(image)
    pose_landmarks = detection_result.pose_landmarks

    nose = pose_landmarks[0][0]
    right_shoulder = pose_landmarks[0][12]
    right_elbow = pose_landmarks[0][14]
    right_wrist = pose_landmarks[0][16]
    left_shoulder = pose_landmarks[0][11]
    left_elbow = pose_landmarks[0][13]
    left_wrist = pose_landmarks[0][15]
    left_hip = pose_landmarks[0][23]
    left_knee = pose_landmarks[0][25]
    left_ankle = pose_landmarks[0][27]
    right_hip = pose_landmarks[0][24]
    right_knee = pose_landmarks[0][26]
    right_ankle = pose_landmarks[0][28]

    right_hand = calculate_distance(right_shoulder.x*image.width, right_shoulder.y*image.height, right_elbow.x*image.width, right_elbow.y*image.height) + calculate_distance(right_elbow.x*image.width, right_elbow.y*image.height, right_wrist.x*image.width, right_wrist.y*image.height)

    left_hand = calculate_distance(left_shoulder.x*image.width, left_shoulder.y*image.height, left_elbow.x*image.width, left_elbow.y*image.height) + calculate_distance(left_elbow.x*image.width, left_elbow.y*image.height, left_wrist.x*image.width, left_wrist.y*image.height)

    right_foot = calculate_distance(right_hip.x*image.width, right_hip.y*image.height, right_knee.x*image.width, right_knee.y*image.height) + calculate_distance(right_knee.x*image.width, right_knee.y*image.height, right_ankle.x*image.width, right_ankle.y*image.height)

    left_foot = calculate_distance(left_hip.x*image.width, left_hip.y*image.height, left_knee.x*image.width, left_knee.y*image.height) + calculate_distance(left_knee.x*image.width, left_knee.y*image.height, left_ankle.x*image.width, left_ankle.y*image.height)

    coords = np.array([
        [nose.x*image.width,nose.y*image.height],
        [right_shoulder.x*image.width,right_shoulder.y*image.height],
        [left_shoulder.x*image.width,left_shoulder.y*image.height],
        [right_elbow.x*image.width,right_elbow.y*image.height],
        [left_elbow.x*image.width,left_elbow.y*image.height],
        [right_wrist.x*image.width,right_wrist.y*image.height],
        [left_wrist.x*image.width,left_wrist.y*image.height],
        [right_hip.x*image.width,right_hip.y*image.height],
        [left_hip.x*image.width,left_hip.y*image.height],
        [right_knee.x*image.width,right_knee.y*image.height],
        [left_knee.x*image.width,left_knee.y*image.height],
        [right_ankle.x*image.width,right_ankle.y*image.height],
        [left_ankle.x*image.width,left_ankle.y*image.height],
        ])

    return coords, right_foot, left_foot

#===============================================================
  def masking(self, input_point, input_label):
    everything_results = model(self.path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(self.path, everything_results, device=DEVICE)
    mask = prompt_process.point_prompt(input_point, input_label)
    return mask

def tarik_garis(mask, pointFrom_mediapipe):
  indices = np.where(mask > 0)
  indices_x = indices[2]
  indices_y = indices[1]

    # Cari nilai indices_x yang sama dengan nilai pointFrom_mediapipe indeks ke-0
  val_poin_pose = int(pointFrom_mediapipe[0])  # Dapatkan nilai di indeks pertama dari pointFrom_mediapipe
  matching_indices = [index for index, value in enumerate(indices_x) if value == val_poin_pose]  # Dapatkan indeks dari nilai val_poin_pose di indices_x
  matching_indices_y_values = [indices_y[index] for index in matching_indices]  # Ambil nilai indices_y yang memiliki indeks yang sama dengan indices_x yang cocok

  print(indices)
  print(indices_x)
  print(indices_y)
  #print("Nilai indices_x yang cocok dengan,", int(pointFrom_mediapipe[0]), "adalah", matching_indices)
  #print("Nilai indices_y dengan indeks yang sama:", matching_indices_y_values)
  #print(indices[1])
  #print(pointFrom_mediapipe)

  # Temukan nilai maksimum dari daftar matching_indices_y_values
  if matching_indices_y_values:
      nilai_maksimum = max(matching_indices_y_values)
      nilai_minimum = min(matching_indices_y_values)
  else:
      print("Tidak ada nilai yang cocok")

  return abs(nilai_maksimum-nilai_minimum)

#=================================Deteksi Koin====================================
def detect(img):
  model = torch.hub.load('.', 'custom', path='model/best.pt', source='local', force_reload=True)
  # Image
  # Inference
  results = model(img)

  # Results, change the flowing to: results.show()
  df = results.pandas().xyxy[0]  # or .show(), .save(), .crop(), .pandas(), etc
  df = df[df['confidence'] > 0.6]
  df['x'] = df['xmax'] - df['xmin']
  df['y'] = df['ymax'] - df['ymin']
  df['x_tengah'] = (df['xmin'] + df['xmax']) / 2
  df['y_tengah'] = (df['ymin'] + df['ymax']) / 2
  df = df.sort_values('x')
  df = df.reset_index()
  x = df['x_tengah'][0]
  y = df['y_tengah'][0]
  # print(df)
  coords = np.array([[x,y]])
  lists = [df['xmin'][0], df['ymin'][0], df['xmax'][0], df['ymax'][0]]
  width = lists[2] - lists[0]
  height = lists[3] - lists[1]
  if width > height:
      width = height
  else:
      pass

  return coords, lists, width

#ambil path foto
image = FastSam('images/bayi-up.jpg')
coords, right_foot, left_foot = image.get_input_point()
coords = coords.astype(int)

#input_label
input_label_kepala = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
input_label_paha = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
input_label_lengan = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# all masking
masking_kepala = image.masking(coords, input_label_kepala)
masking_lengan = image.masking(coords, input_label_lengan)
masking_paha = image.masking(coords, input_label_paha)

#ambil koordinat
x_rightshoulder = coords[1][0]
y_rightshoulder = coords[1][1]
x_leftshoulder = coords[2][0]
y_leftshoulder = coords[2][1]
x_righthip = coords[5][0]
y_righthip = coords[5][1]
x_lefthip = coords[6][0]
y_lefthip = coords[6][1]

def coef(img):
  real_coin_size = 2.7
  coin_size = detect(img)[2]
  coef = real_coin_size / coin_size
  return coef

#mencari koefisien koin
koef_koin1 = coef('images/bayi-up.jpg')

# all masking
masking_kepala = image.masking(coords, input_label_kepala)
masking_lengan = image.masking(coords, input_label_lengan)
masking_paha = image.masking(coords, input_label_paha)

# tarik garis
garis_kepala = tarik_garis(masking_kepala, coords[0])*koef_koin1
garis_kepala2 = garis_kepala*0.6

garis_lengan = tarik_garis(masking_lengan, coords[5])*koef_koin1
garis_lengan2 = garis_lengan*0.6

garis_paha = tarik_garis(masking_paha, coords[10])*koef_koin1
garis_paha2 = garis_paha*0.01

#perhitungan lingkar
lingkar_kepala = perhitungan_elips(garis_kepala, garis_kepala2)
lingkar_lengan = perhitungan_elips(garis_lengan, garis_lengan2)
lingkar_paha = perhitungan_elips(garis_paha, garis_paha2)

tinggi_dada = garis_kepala*1.2
tinggi_perut = garis_kepala*1.25

lebar_dada=abs(y_rightshoulder-y_leftshoulder)*koef_koin1
lebar_pinggang=(abs(y_righthip-y_lefthip)-0.4*abs(y_righthip-y_lefthip))*koef_koin1

# perhitungan lingkar
lingkar_dada = perhitungan_elips(tinggi_dada, lebar_dada)
lingkar_perut = perhitungan_elips(tinggi_perut, lebar_pinggang)

print(lingkar_dada, lingkar_perut, lingkar_kepala, lingkar_lengan, lingkar_paha)