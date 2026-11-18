import time
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# =====================================================================
# 1. WRAPPER FUNCTIONS (THE "PLUG-AND-PLAY" STUBS)
# =====================================================================

def run_baseline(clean_image):
    """
    Baseline: Feeds a clean image into the original captioning model.
    Returns: (generated_caption, execution_time_seconds)
    """
    start_time = time.perf_counter()
    
    # TODO: Replace this dummy string with the real Baseline model inference
    generated_caption = "một con mèo đang ngồi trên ghế"
    
    elapsed_time = time.perf_counter() - start_time
    return generated_caption, elapsed_time


def run_pipeline_1_dncnn(noisy_image):
    """
    Pipeline 1: Cleans the noisy image using DnCNN, then generates a caption.
    Returns: (denoised_image, generated_caption, execution_time_seconds)
    """
    start_time = time.perf_counter()
    
    # TODO: Step A - Pass noisy_image through DnCNN to get denoised_image
    # For now, we return a dummy image (same shape as input)
    denoised_image = noisy_image.copy() 
    
    # TODO: Step B - Pass denoised_image through the captioning model
    generated_caption = "con mèo ngồi trên chiếc ghế"
    
    elapsed_time = time.perf_counter() - start_time
    return denoised_image, generated_caption, elapsed_time


def run_pipeline_2_end2end(noisy_image):
    """
    Pipeline 2: Feeds noisy image directly into the fine-tuned model.
    Returns: (generated_caption, execution_time_seconds)
    """
    start_time = time.perf_counter()
    
    # TODO: Replace with real End-to-End fine-tuned model inference
    generated_caption = "một chú mèo đang nằm trên ghế"
    
    elapsed_time = time.perf_counter() - start_time
    return generated_caption, elapsed_time


# =====================================================================
# 2. METRIC CALCULATION HELPERS
# =====================================================================

def calculate_bleu_scores(reference_caption, candidate_caption):
    """
    Compares the AI's caption against the human reference caption.
    Returns a dictionary of BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores.
    """
    # Split sentences into lists of words (tokens)
    ref_tokens = [reference_caption.lower().split()]
    cand_tokens = candidate_caption.lower().split()
    
    # Smoothing prevents the score from dropping to 0 on short sentences
    smooth = SmoothingFunction().method1
    
    bleu_1 = sentence_bleu(ref_tokens, cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu_2 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu_3 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu_4 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    return {
        "BLEU-1": round(bleu_1, 4),
        "BLEU-2": round(bleu_2, 4),
        "BLEU-3": round(bleu_3, 4),
        "BLEU-4": round(bleu_4, 4)
    }


def calculate_image_quality(clean_img, denoised_img):
    """
    Measures how well DnCNN cleaned the image compared to the original clean image.
    Assumes images are NumPy arrays with pixel values in [0, 255] or [0, 1].
    """
    # data_range=1.0 if normalized [0,1], or 255 if standard image pixels
    psnr_val = psnr(clean_img, denoised_img, data_range=1.0)
    ssim_val = ssim(clean_img, denoised_img, channel_axis=-1, data_range=1.0)
    
    return round(psnr_val, 2), round(ssim_val, 4)


# =====================================================================
# 3. MAIN BENCHMARK LOOP
# =====================================================================

def run_benchmark():
    print("Starting Benchmark Evaluation...")
    
    # Dummy test dataset (Replace this with your actual DataLoader from dataset.py)
    # Format: (image_id, clean_image_array, noisy_image_array, human_caption)
    dummy_test_data = [
        ("img_001.jpg", np.ones((224, 224, 3)), np.ones((224, 224, 3)) * 0.9, "một con mèo đang ngồi trên ghế"),
        ("img_002.jpg", np.ones((224, 224, 3)), np.ones((224, 224, 3)) * 0.85, "người đàn ông đang đi bộ trên đường")
    ]
    
    all_results = []

    for img_id, clean_img, noisy_img, ref_caption in dummy_test_data:
        # 1. Run all three pipelines
        base_cap, base_time = run_baseline(clean_img)
        p1_denoised, p1_cap, p1_time = run_pipeline_1_dncnn(noisy_img)
        p2_cap, p2_time = run_pipeline_2_end2end(noisy_img)
        
        # 2. Grade the text outputs
        base_bleu = calculate_bleu_scores(ref_caption, base_cap)
        p1_bleu = calculate_bleu_scores(ref_caption, p1_cap)
        p2_bleu = calculate_bleu_scores(ref_caption, p2_cap)
        
        # 3. Grade the image cleaning (Only Pipeline 1 does image denoising)
        p1_psnr, p1_ssim = calculate_image_quality(clean_img, p1_denoised)
        
        # 4. Record everything into a single row
        row = {
            "Image_ID": img_id,
            "Reference_Caption": ref_caption,
            
            # Baseline Results
            "Base_Caption": base_cap,
            "Base_BLEU4": base_bleu["BLEU-4"],
            "Base_Time_s": round(base_time, 4),
            
            # Pipeline 1 Results
            "P1_Caption": p1_cap,
            "P1_PSNR": p1_psnr,
            "P1_SSIM": p1_ssim,
            "P1_BLEU4": p1_bleu["BLEU-4"],
            "P1_Time_s": round(p1_time, 4),
            
            # Pipeline 2 Results
            "P2_Caption": p2_cap,
            "P2_BLEU4": p2_bleu["BLEU-4"],
            "P2_Time_s": round(p2_time, 4),
        }
        all_results.append(row)

    # Convert to a Pandas table and save to CSV
    df = pd.DataFrame(all_results)
    output_path = "data/processed/benchmark_comparison.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"Done! Results saved to {output_path}")
    print("\nPreview of Average Scores:")
    print(df[["Base_BLEU4", "P1_BLEU4", "P2_BLEU4", "P1_PSNR", "P1_SSIM"]].mean())


if __name__ == "__main__":
    run_benchmark()
