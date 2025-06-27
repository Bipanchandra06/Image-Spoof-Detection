import os
import shutil
import cv2
import numpy as np
import pandas as pd
import random
from glob import glob
from tqdm import tqdm 
from skimage.feature import local_binary_pattern
from skimage.restoration import denoise_tv_chambolle
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Input Data Paths
LFW_RAW_DIR = "R:/DA221M/Datasets/Real/archive/lfw-funneled/lfw_funneled" # Input for real face extraction
SPOOF_VIDEO_DIR = "R:/DA221M/Datasets/Spoof/Videos" # Input videos for spoof frames

# 2. Intermediate/Output Paths (Organized)
BASE_OUTPUT_DIR = "anti_spoofing_data_simple" 
REAL_EXTRACTED_DIR = os.path.join(BASE_OUTPUT_DIR, "1_real_extracted")
SPOOF_FRAMES_DIR = os.path.join(BASE_OUTPUT_DIR, "2_spoof_frames")
SPOOF_AUGMENTED_DIR = os.path.join(BASE_OUTPUT_DIR, "3_spoof_augmented")
REAL_PREPROCESSED_DIR = os.path.join(BASE_OUTPUT_DIR, "4_real_preprocessed")
SPOOF_PREPROCESSED_DIR = os.path.join(BASE_OUTPUT_DIR, "4_spoof_preprocessed")
LBP_FEATURES_CSV = os.path.join(BASE_OUTPUT_DIR, "5_lbp_features.csv")
LBP_FEATURES_SHUFFLED_CSV = os.path.join(BASE_OUTPUT_DIR, "6_lbp_features_shuffled.csv")

# 3. Processing Parameters
FRAME_EXTRACTION_INTERVAL = 10
TARGET_IMG_SIZE = (128, 128) 
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
SVM_TEST_SIZE = 0.2
RANDOM_STATE = 42 

# --- Helper Functions ---

def create_dir_if_not_exists(directory):
    os.makedirs(directory, exist_ok=True)

# --- Stage 1: Data Extraction ---

def flatten_real_images(input_root_folder, output_folder):
    print(f"Starting extraction of real images from '{input_root_folder}'...")
    create_dir_if_not_exists(output_folder)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
    extracted_count = 0
    for subdir, _, files in os.walk(input_root_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                src_path = os.path.join(subdir, file)
                dest_file_name = os.path.basename(src_path)
                dest_path = os.path.join(output_folder, dest_file_name)

                count = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(dest_file_name)
                    dest_path = os.path.join(output_folder, f"{name}_{count}{ext}")
                    count += 1
                    if count > 10: 
                         print(f"WARNING: Too many duplicates for {dest_file_name}, skipping further checks.")
                         break

                if not os.path.exists(dest_path):
                    try:
                        shutil.copy2(src_path, dest_path)
                        extracted_count += 1
                    except Exception as e:
                        print(f"ERROR: Failed to copy {src_path}: {e}")

    print(f"Extraction complete! {extracted_count} real images saved in: {output_folder}")
    return extracted_count

def extract_frames_from_videos(video_dir, output_dir, frame_interval):
    print(f"Starting frame extraction from videos in '{video_dir}'...")
    create_dir_if_not_exists(output_dir)
    total_frames_saved = 0
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".mov", ".avi"))]

    if not video_files:
        print(f"WARNING: No video files found in {video_dir}")
        return 0

    for video_file in tqdm(video_files, desc="Extracting Spoof Frames"):
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"WARNING: Could not open video file: {video_path}")
            continue

        frame_count = 0
        saved_frame_count = 0
        video_base_name = os.path.splitext(video_file)[0]

        while True:
            ret, frame = cap.read()
            if not ret:
                break 

            if frame_count % frame_interval == 0:
                frame_filename = f"{video_base_name}_frame_{saved_frame_count}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                try:
                    cv2.imwrite(frame_path, frame)
                    saved_frame_count += 1
                except Exception as e:
                    print(f"ERROR: Failed to write frame {frame_filename}: {e}")

            frame_count += 1

        cap.release()
        total_frames_saved += saved_frame_count

    print(f"Frame extraction complete! {total_frames_saved} frames saved in: {output_dir}")
    return total_frames_saved

# --- Stage 2: Data Augmentation (for Spoof data) ---

def augment_spoof_images(spoof_input_dir, spoof_output_dir, target_count):
    print("Starting spoof image augmentation...")
    create_dir_if_not_exists(spoof_output_dir)

    existing_spoof_paths = glob(os.path.join(spoof_input_dir, "*.jpg"))
    current_spoof_count = len(existing_spoof_paths)
    print(f"Found {current_spoof_count} existing spoof frames.")

    if not existing_spoof_paths:
        print(f"ERROR: No spoof frames found in {spoof_input_dir} to augment.")
        return 0

    print(f"Copying {current_spoof_count} existing frames to {spoof_output_dir}...")
    for img_path in tqdm(existing_spoof_paths, desc="Copying existing spoof frames"):
         try:
             shutil.copy2(img_path, os.path.join(spoof_output_dir, os.path.basename(img_path)))
         except Exception as e:
             print(f"ERROR: Could not copy {img_path}: {e}")


    augmentation_needed = max(0, target_count - current_spoof_count)
    print(f"Target count (based on real images): {target_count}. Need to generate {augmentation_needed} augmented images.")

    if augmentation_needed == 0:
        print("No augmentation needed.")
        create_dir_if_not_exists(spoof_output_dir)
        return current_spoof_count 

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ])

    augmented_count = 0
    pbar = tqdm(total=augmentation_needed, desc="Generating augmented images")
    while augmented_count < augmentation_needed:
        img_path = random.choice(existing_spoof_paths) 
        img = cv2.imread(img_path)

        if img is None:
            print(f"WARNING: Could not read image {img_path} for augmentation. Skipping.")
            continue

        try:
            augmented = transform(image=img)['image']
            save_path = os.path.join(spoof_output_dir, f"aug_{augmented_count:06d}.jpg") 
            cv2.imwrite(save_path, augmented)
            augmented_count += 1
            pbar.update(1)
        except Exception as e:
             print(f"ERROR: Augmentation or saving failed for {img_path}: {e}")

    pbar.close()
    final_count = len(glob(os.path.join(spoof_output_dir, "*.jpg")))
    print(f"Augmentation complete! {augmented_count} images generated. Total spoof images in output folder: {final_count}")
    return final_count

# --- Stage 3: Preprocessing ---

def preprocess_image(img_path, output_dir, img_size):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: Could not read image: {img_path}. Skipping.")
            return False

        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Apply LTV denoising
        denoised = denoise_tv_chambolle(gray, weight=0.1, channel_axis=None) 
        denoised_uint8 = np.clip(denoised * 255, 0, 255).astype(np.uint8)

        # 3. Resize the image
        resized = cv2.resize(denoised_uint8, img_size, interpolation=cv2.INTER_AREA) 

        # 4. Apply CLAHE 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(resized)

        # 5. Normalization 
        processed_img = enhanced

        # Save processed image
        base_filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, base_filename)
        cv2.imwrite(save_path, processed_img) 
        return True

    except Exception as e:
        print(f"ERROR: Error processing image {img_path}: {e}")
        return False

def preprocess_folder(input_folder, output_folder, img_size):
    print(f"Starting preprocessing for folder: '{input_folder}'...")
    create_dir_if_not_exists(output_folder)
    image_files = glob(os.path.join(input_folder, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    processed_count = 0
    skipped_count = 0

    if not image_files:
        print(f"WARNING: No image files found in {input_folder}")
        return

    for img_path in tqdm(image_files, desc=f"Preprocessing {os.path.basename(input_folder)}"):
        if preprocess_image(img_path, output_folder, img_size):
            processed_count += 1
        else:
            skipped_count += 1

    print(f"✅ Preprocessing complete for {input_folder}. Processed: {processed_count}, Skipped/Errors: {skipped_count}")

# --- Stage 4: LBP Feature Extraction ---

def extract_single_lbp(image_path, n_points, radius):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"WARNING: Could not load image {image_path} for LBP. Skipping.")
            return None

        lbp = local_binary_pattern(img, n_points, radius, method="uniform")

        n_bins = int(n_points + 2) 
        hist, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, n_bins + 1), 
                               range=(0, n_bins),
                               density=True) 
        return hist

    except Exception as e:
        print(f"ERROR: Error extracting LBP for {image_path}: {e}")
        return None

def extract_lbp_features_to_csv(real_folder, fake_folder, output_csv, n_points, radius):
    data = []
    labels = []
    real_files = glob(os.path.join(real_folder, "*.*"))
    real_files = [f for f in real_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    print(f"Found {len(real_files)} potential real images in: {real_folder}")
    if not real_files: print(f"WARNING: No real image files found in {real_folder}")

    for img_path in tqdm(real_files, desc="Processing Real Faces (LBP)"):
        features = extract_single_lbp(img_path, n_points, radius)
        if features is not None:
            data.append(features)
            labels.append(0)

    fake_files = glob(os.path.join(fake_folder, "*.*"))
    fake_files = [f for f in fake_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    print(f"Found {len(fake_files)} potential fake images in: {fake_folder}")
    if not fake_files: print(f"WARNING: No fake image files found in {fake_folder}")

    for img_path in tqdm(fake_files, desc="Processing Fake Faces (LBP)"):
        features = extract_single_lbp(img_path, n_points, radius)
        if features is not None:
            data.append(features)
            labels.append(1)

    if not data:
        print("ERROR: No features were extracted. Cannot create CSV.")
        return False


    try:
        feature_dim = len(data[0])
        col_names = [f"feat_{i}" for i in range(feature_dim)]
        df = pd.DataFrame(data, columns=col_names)
        df["label"] = labels 
        df.to_csv(output_csv, index=False) 
        print(f"LBP Feature Extraction Complete! Data saved to '{output_csv}'. Shape: {df.shape}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to create or save DataFrame to CSV: {e}")
        return False

# --- Stage 5: Data Shuffling ---

def shuffle_csv(input_csv_path, output_csv_path, random_state):
    """Shuffles the rows of a CSV file using pandas."""
    print(f"Shuffling dataset: '{input_csv_path}'...")
    try:
        df = pd.read_csv(input_csv_path)
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        df_shuffled.to_csv(output_csv_path, index=False)
        print(f"✅ Dataset shuffled and saved to '{output_csv_path}'.")
        return True
    except FileNotFoundError:
        print(f"ERROR: Input CSV not found: {input_csv_path}")
        return False
    except Exception as e:
        print(f"ERROR: Error shuffling CSV: {e}")
        return False

# --- Stage 6: SVM Training and Evaluation ---

def train_evaluate_svm(feature_csv_path, test_size, random_state):
    """Trains an SVM classifier and evaluates its performance using sklearn."""
    print(f"Starting SVM training using features from: '{feature_csv_path}'...")
    try:
        df = pd.read_csv(feature_csv_path)
    except FileNotFoundError:
        print(f"ERROR: Feature CSV not found: {feature_csv_path}")
        return False
    except pd.errors.EmptyDataError:
         print(f"ERROR: Feature CSV is empty: {feature_csv_path}")
         return False


    if df.empty:
        print("ERROR: Feature DataFrame is empty after loading. Cannot train model.")
        return False

    # Separate features (X) and labels (y) using iloc as in svm.py
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values   

    if X.shape[0] == 0 or X.shape[1] == 0:
         print(f"ERROR: Invalid data shape after loading CSV. X shape: {X.shape}. Cannot train.")
         return False
    if len(np.unique(y)) < 2:
        print(f"ERROR: Only found {len(np.unique(y))} classes in the data. Need at least 2 for classification. Labels found: {np.unique(y)}")
        return False


    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features per sample.")

    # Split into training and testing sets using sklearn.model_selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")

    # Training the SVM classifier 
    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)
    print("Training SVM model (kernel=rbf, C=1.0, gamma=scale)...")
    try:
        svm_model.fit(X_train, y_train)
    except Exception as e:
        print(f"ERROR: SVM training failed: {e}")
        return False

    print("SVM training complete.")

    print("Making predictions on the test set...")
    y_pred = svm_model.predict(X_test)

    # Evaluate the model using sklearn.metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Real (0)', 'Spoof (1)'], zero_division=0)

    # Print results
    print("\n" + "="*30 + " SVM Evaluation Results " + "="*30)
    print(f"✅ Model Accuracy: {accuracy:.4f}")
    print("\n🔹 Confusion Matrix:\n", conf_matrix)
    print("\n🔹 Classification Report:\n", report)
    print("="*80)


    return True

# --- Main Execution Pipeline ---

if __name__ == "__main__":
    create_dir_if_not_exists(BASE_OUTPUT_DIR)

    pipeline_successful = True 

    # --- Step 1: Extract Real Face Images ---
    print("\n--- Step 1: Extracting Real Images ---")
    if not os.path.isdir(LFW_RAW_DIR):
         print(f"ERROR: Real image source directory not found: {LFW_RAW_DIR}. Checking target...")
         if os.path.isdir(REAL_EXTRACTED_DIR) and glob(os.path.join(REAL_EXTRACTED_DIR, '*.*')):
              real_img_count = len(glob(os.path.join(REAL_EXTRACTED_DIR, '*.*')))
              print(f"WARNING: Using {real_img_count} existing images from {REAL_EXTRACTED_DIR}")
         else:
              print(f"ERROR: Cannot find source {LFW_RAW_DIR} or existing images in {REAL_EXTRACTED_DIR}. Aborting.")
              pipeline_successful = False
    else:
        real_img_count = flatten_real_images(LFW_RAW_DIR, REAL_EXTRACTED_DIR)
        if real_img_count == 0:
            print("ERROR: Failed to extract any real images. Check paths and permissions. Aborting.")
            pipeline_successful = False

    # --- Step 2: Extract Spoof Frames ---
    if pipeline_successful:
        print("\n--- Step 2: Extracting Spoof Frames from Videos ---")
        if not os.path.isdir(SPOOF_VIDEO_DIR):
            print(f"ERROR: Spoof video directory not found: {SPOOF_VIDEO_DIR}. Checking target...")
            if os.path.isdir(SPOOF_FRAMES_DIR) and glob(os.path.join(SPOOF_FRAMES_DIR, '*.*')):
                 print(f"WARNING: Using existing frames from {SPOOF_FRAMES_DIR}")
            elif not os.path.isdir(SPOOF_FRAMES_DIR) or not glob(os.path.join(SPOOF_FRAMES_DIR, '*.*')):
                 print("ERROR: No existing spoof frames found either. Aborting.")
                 pipeline_successful = False
        else:
            extracted_frame_count = extract_frames_from_videos(SPOOF_VIDEO_DIR, SPOOF_FRAMES_DIR, FRAME_EXTRACTION_INTERVAL)
            if not glob(os.path.join(SPOOF_FRAMES_DIR, '*.*')):
                 print("ERROR: Failed to extract any spoof frames, and none existed previously. Aborting.")
                 pipeline_successful = False


    # --- Step 3: Augment Spoof Images ---
    if pipeline_successful:
        print("\n--- Step 3: Augmenting Spoof Images (if needed) ---")
        total_spoof_count = augment_spoof_images(SPOOF_FRAMES_DIR, SPOOF_AUGMENTED_DIR, real_img_count)
        if total_spoof_count == 0:
            print("ERROR: No spoof images available after augmentation step. Aborting.")
            pipeline_successful = False

    # --- Step 4: Preprocess Images ---
    if pipeline_successful:
        print("\n--- Step 4: Preprocessing Real and Spoof Images ---")
        print("Preprocessing Real Images...")
        if not glob(os.path.join(REAL_EXTRACTED_DIR, '*.*')):
             print(f"ERROR: No images found in {REAL_EXTRACTED_DIR} to preprocess.")
             pipeline_successful = False
        else:
            preprocess_folder(REAL_EXTRACTED_DIR, REAL_PREPROCESSED_DIR, TARGET_IMG_SIZE)

    if pipeline_successful:
        print("Preprocessing Spoof Images (including augmented)...")
        if not glob(os.path.join(SPOOF_AUGMENTED_DIR, '*.*')):
             print(f"ERROR: No images found in {SPOOF_AUGMENTED_DIR} to preprocess.")
             pipeline_successful = False
        else:

            preprocess_folder(SPOOF_AUGMENTED_DIR, SPOOF_PREPROCESSED_DIR, TARGET_IMG_SIZE)

    # --- Step 5: Extract LBP Features ---
    if pipeline_successful:
        print("\n--- Step 5: Extracting LBP Features ---")
        if not os.path.isdir(REAL_PREPROCESSED_DIR) or not glob(os.path.join(REAL_PREPROCESSED_DIR, '*.*')):
             print(f"ERROR: Real preprocessed directory {REAL_PREPROCESSED_DIR} is empty or missing.")
             pipeline_successful = False
        elif not os.path.isdir(SPOOF_PREPROCESSED_DIR) or not glob(os.path.join(SPOOF_PREPROCESSED_DIR, '*.*')):
             print(f"ERROR: Spoof preprocessed directory {SPOOF_PREPROCESSED_DIR} is empty or missing.")
             pipeline_successful = False
        else:
            success_lbp = extract_lbp_features_to_csv(
                REAL_PREPROCESSED_DIR,
                SPOOF_PREPROCESSED_DIR,
                LBP_FEATURES_CSV,
                LBP_N_POINTS,
                LBP_RADIUS
            )
            if not success_lbp:
                print("ERROR: LBP Feature extraction failed.")
                pipeline_successful = False

    # --- Step 6: Shuffle Features ---
    if pipeline_successful:
        print("\n--- Step 6: Shuffling Feature Dataset ---")
        success_shuffle = shuffle_csv(LBP_FEATURES_CSV, LBP_FEATURES_SHUFFLED_CSV, RANDOM_STATE)
        if not success_shuffle:
            print("ERROR: Dataset shuffling failed.")
            pipeline_successful = False

    # --- Step 7: Train and Evaluate SVM ---
    if pipeline_successful:
        print("\n--- Step 7: Training and Evaluating SVM Model ---")
        success_svm = train_evaluate_svm(LBP_FEATURES_SHUFFLED_CSV, SVM_TEST_SIZE, RANDOM_STATE)
        if not success_svm:
            print("ERROR: SVM training/evaluation failed.")
            pipeline_successful = False

    # --- Final Status ---
    if pipeline_successful:
        print("\nAnti-Spoofing Pipeline Finished Successfully!")
    else:
        print("\nAnti-Spoofing Pipeline Finished with ERRORS.")