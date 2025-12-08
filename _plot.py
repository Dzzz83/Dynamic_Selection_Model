import matplotlib.pyplot as plt
import re
import glob
import os

# --- Configuration ---
# Where are your log files? ('.' means current folder)
LOG_DIR = 'results' 
# pattern to match your files (e.g., "train_*.log" or just "*.log")
FILE_PATTERN = "train_*.log" 

def parse_log_file(filepath):
    """Parses a single log file and returns lists of metrics."""
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Regex to extract numbers from your specific log format
    pattern = re.compile(r"TrLoss:\s+([\d\.]+)\s+Acc:\s+([\d\.]+).*?ValLoss:\s+([\d\.]+)\s+Acc:\s+([\d\.]+)")
    
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                t_loss, t_acc, v_loss, v_acc = match.groups()
                train_losses.append(float(t_loss))
                train_accs.append(float(t_acc))
                val_losses.append(float(v_loss))
                val_accs.append(float(v_acc))
                
    return train_losses, train_accs, val_losses, val_accs

def create_plot(t_loss, v_loss, t_acc, v_acc, filename):
    """Generates and saves a plot for the given data."""
    epochs = range(1, len(t_loss) + 1)
    
    # Extract clean name for title
    # Handles both timestamped files (_202...) and your new deterministic files
    base = os.path.basename(filename)
    if "_202" in base:
        clean_name = base.split('train_')[-1].split('_202')[0]
    else:
        # Fallback for files without timestamp (like '...dynamic_r0.7.log')
        clean_name = base.replace('.log', '').replace('train_', '')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Loss
    ax1.plot(epochs, t_loss, 'b-', label='Train')
    ax1.plot(epochs, v_loss, 'r--', label='Val')
    ax1.set_title(f'Loss: {clean_name}')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Plot Accuracy
    ax2.plot(epochs, t_acc, 'b-', label='Train')
    ax2.plot(epochs, v_acc, 'r--', label='Val')
    
    max_val = max(v_acc) if len(v_acc) > 0 else 0
    ax2.set_title(f'Accuracy (Best: {max_val:.2f}%): {clean_name}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save file: "train_log.txt" -> "train_log.png"
    save_name = filename.replace('.log', '.png')
    plt.savefig(save_name)
    plt.close() # Close memory to prevent leaks in loops
    print(f"Generated: {save_name}")

if __name__ == "__main__":
    # 1. Find all matching files
    search_path = os.path.join(LOG_DIR, FILE_PATTERN)
    log_files = glob.glob(search_path)
    log_files.sort(key=os.path.getmtime)
    
    print(f"Found {len(log_files)} log files.")

    # 2. Loop through each file
    for log_file in log_files:
        # ✅ NEW: Check if plot exists before processing
        expected_png = log_file.replace('.log', '.png')
        
        if os.path.exists(expected_png):
            print(f"⏭️  [Skipping] Plot already exists: {os.path.basename(expected_png)}")
            continue

        # If not exists, proceed to parse and plot
        print(f"Processing {log_file}...")
        try:
            t_loss, t_acc, v_loss, v_acc = parse_log_file(log_file)
            
            if len(t_loss) > 0:
                create_plot(t_loss, v_loss, t_acc, v_acc, log_file)
            else:
                print(f"   [Skipped] No valid metric data found in {log_file}")
        except Exception as e:
            print(f"   [Error] Could not process {log_file}: {e}")