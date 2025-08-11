#!/usr/bin/env python3
"""
Loss chart generation tool
Usage:
    python plot_loss.py                    # Automatically find the latest results.csv
    python plot_loss.py path/to/results.csv  # Specify csv file path
"""

import sys
import os
from utils import load_config, find_latest_results_csv, plot_training_results

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # If parameter is provided, use the specified CSV file
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"❌ File does not exist: {csv_path}")
            return False
    else:
        # Automatically find the latest results.csv
        config = load_config()
        csv_path = find_latest_results_csv(config)

        if not csv_path:
            print("❌ No results.csv file found")
            print("Please ensure training has been run, or manually specify CSV file path:")
            print("   python plot_loss.py path/to/results.csv")
            return False

        print(f"🔍 Found latest results.csv: {csv_path}")

    # Generate chart
    success = plot_training_results(csv_path)

    if success:
        print("🎉 Loss chart generation completed!")
        # Display save location
        save_path = os.path.join(os.path.dirname(csv_path), 'results.png')
        print(f"📊 Chart save location: {save_path}")
        return True
    else:
        print("❌ Chart generation failed")
        return False

if __name__ == "__main__":
    main()