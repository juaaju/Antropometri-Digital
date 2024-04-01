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

#load model
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "model/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Fungsi Landmark
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

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
    return [y, x, width]

#tarik garis
def tarik_garis(mask, pointFrom_mediapipe):
  indices = np.where(mask > 0)
  indices_x = indices[2]
  indices_y = indices[1]

    # Cari nilai indices_x yang sama dengan nilai pointFrom_mediapipe indeks ke-0
  val_poin_pose = int(pointFrom_mediapipe[0])  # Dapatkan nilai di indeks pertama dari pointFrom_mediapipe
  matching_indices = [index for index, value in enumerate(indices_x) if value == val_poin_pose]  # Dapatkan indeks dari nilai val_poin_pose di indices_x
  matching_indices_y_values = [indices_y[index] for index in matching_indices]  # Ambil nilai indices_y yang memiliki indeks yang sama dengan indices_x yang cocok

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

class FastSam():
  def __init__(self, IMAGE_PATH):
    self.path = IMAGE_PATH
    self.image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    predictor.set_image(self.image)

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


    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255

    return coords, right_foot, left_foot

#===============================================================================
  def munculkan_foto(self):
    plt.figure(figsize=(10, 10))
    plt.imshow(self.image)
    plt.title(f"Hasil Gambar")
    plt.axis('on')
    plt.show()

  def munculkan_titik_mediapipe(self, coords, input_label):
    plt.figure(figsize=(10, 10))
    plt.imshow(self.image)
    show_points(coords, input_label, plt.gca())
    plt.title(f"Titik pada Tubuh")
    plt.axis('on')
    plt.show()

  def cek_masking(self, masks):
    plt.figure(figsize=(10, 10))
    plt.imshow(self.image)
    show_mask(masks, plt.gca())
    plt.title(f"Titik pada Tubuh")
    plt.axis('on')
    plt.show()

# Masking ====================================================================
  def masking(self, input_point, input_label):
    masks, scores, logits = predictor.predict(
      point_coords=input_point,
      point_labels=input_label,
      multimask_output=True,
  )

    mask_input = logits[np.argmax(scores), :, :]

    masks, _, _ = predictor.predict(
      point_coords=input_point,
      point_labels=input_label,
      mask_input=mask_input[None, :, :],
      multimask_output=False,
  )
    return masks

input_label_kepala = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
input_label_paha = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
input_label_lengan = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

input_label_kepala2 = np.array([1, 0, 0, 0, 0, 0, 0])
input_label_paha2 = np.array([0, 0, 0, 0, 0, 1, 0])
input_label_lengan2 = np.array([0, 0, 1, 0, 0, 0, 0])
input_label_badan = np.array([0, 1, 0, 0, 0, 0, 0])

def coef(image_path):
    real_coin_size = 2.7
    coin_size = detect(image_path)[2]
    coef = real_coin_size / coin_size
    return coef

def proses_pertama1(image_path):
    print("First Processing...")
    image = FastSam(image_path)
    coords, right_foot, left_foot = image.get_input_point()

    x_rightshoulder = coords[1][0]
    y_rightshoulder = coords[1][1]
    x_leftshoulder = coords[2][0]
    y_leftshoulder = coords[2][1]
    x_righthip = coords[5][0]
    y_righthip = coords[5][1]
    x_lefthip = coords[6][0]
    y_lefthip = coords[6][1]

    koef_koin1 = coef(image_path)

    # masking
    print("First Masking...")
    masking_kepala = image.masking(coords, input_label_kepala)
    masking_lengan = image.masking(coords, input_label_lengan)
    masking_paha = image.masking(coords, input_label_paha)

    # tarik garis
    garis_kepala = tarik_garis(masking_kepala, coords[0])*koef_koin1
    garis_lengan = tarik_garis(masking_lengan, coords[3])*koef_koin1
    garis_paha = tarik_garis(masking_paha, coords[10])*koef_koin1

    lebar_dada=abs(y_rightshoulder-y_leftshoulder)*koef_koin1
    lebar_pinggang=(abs(y_righthip-y_lefthip)-0.4*abs(y_righthip-y_lefthip))*koef_koin1

    #menghitung total panjang badan
    panjang_kepala = calculate_bbox_from_mask(masking_kepala)[2]*koef_koin1
    panjang_badan = abs(x_leftshoulder-x_lefthip)*koef_koin1
    panjang_kaki = (right_foot + left_foot)*koef_koin1/2
    total_badan = panjang_kepala + panjang_badan + panjang_kaki

    return [garis_kepala, garis_lengan, garis_paha, total_badan, lebar_dada, lebar_pinggang]

def proses_kedua2(image_path):
    print("Second Processing...")
    image = FastSam(image_path)
    coords = image.get_input_point()[0]

    input_poin = np.array([
    [coords[0][0], coords[0][1]],
    [coords[2][0], coords[2][1]],
    [coords[4][0], coords[4][1]],
    [coords[6][0], coords[6][1]],
    [coords[8][0], coords[8][1]],
    [coords[10][0], coords[10][1]],
    [coords[12][0], coords[12][1]],
    ])

    koef_koin2 = coef(image_path)

    # masking
    print("Second Masking...")
    masking_kepala = image.masking(input_poin, input_label_kepala2)
    masking_lengan = image.masking(input_poin, input_label_lengan2)
    masking_paha = image.masking(input_poin, input_label_paha2)

    garis_kepala = tarik_garis(masking_kepala, input_poin[0])*koef_koin2
    garis_lengan = tarik_garis(masking_lengan, input_poin[2])*koef_koin2
    garis_paha = tarik_garis(masking_paha, input_poin[5])*koef_koin2

    tinggi_dada = garis_kepala*0.8
    tinggi_perut = garis_kepala*0.9

    return [garis_kepala, garis_lengan, garis_paha, tinggi_dada, tinggi_perut]

def all_params(image1, image2):
    proses_pertama = proses_pertama1(image1)
    print("1 Measurement...")
    garis_kepala1 = proses_pertama[0]
    garis_lengan1 = proses_pertama[1]
    garis_paha1 = proses_pertama[2]
    total_badan = proses_pertama[3]
    lebar_dada = proses_pertama[4]
    lebar_perut = proses_pertama[5]

    proses_kedua = proses_kedua2(image2)
    print("2 Measurement...")
    garis_kepala2 = proses_kedua[0]
    garis_lengan2 = proses_kedua[1]
    garis_paha2 = proses_kedua[2]
    tinggi_dada = proses_kedua[3]
    tinggi_perut = proses_kedua[4]

    lingkar_kepala = perhitungan_elips(garis_kepala1, garis_kepala2)
    lingkar_lengan = perhitungan_elips(garis_lengan1, garis_lengan2)
    lingkar_paha = perhitungan_elips(garis_paha1, garis_paha2)
    lingkar_dada = perhitungan_elips(tinggi_dada, lebar_dada)
    lingkar_perut = perhitungan_elips(tinggi_perut, lebar_perut)

    return lingkar_kepala, lingkar_lengan, lingkar_paha, lingkar_dada, lingkar_perut, total_badan

print(all_params('baby5-up.jpeg', 'baby5-side.jpeg'))
#print(all_params('images/bayi-up1.jpeg', 'images/bayi-side.jpg'))
print(all_params('images/bayi-up.jpeg', 'images/baby5-side2.jpeg'))