import os
import argparse

def print_tree(startpath, prefix=""):
    """Recursively print the directory tree."""
    try:
        entries = sorted(os.listdir(startpath))
    except PermissionError:
        print(prefix + "└── [Permission Denied]")
        return

    for count, entry in enumerate(entries):
        path = os.path.join(startpath, entry)
        connector = "└── " if count == len(entries) - 1 else "├── "
        print(prefix + connector + entry)
        if os.path.isdir(path):
            extension = "    " if count == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)

def print_summary(summary_file):
    """Print contents of the summary file if it exists."""
    if os.path.isfile(summary_file):
        print("\n--- Project Achievements & Summary ---")
        with open(summary_file, 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print("\nNo summary file found. Create one (e.g., achievements.txt) with your project milestones.")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a project directory tree and display project achievements/summary."
    )
    parser.add_argument(
        "project_path", help="Path to the project root folder", nargs="?", default="."
    )
    parser.add_argument(
        "--summary", help="Filename of the summary/achievements file", default="achievements.txt"
    )
    
    args = parser.parse_args()
    
    project_path = os.path.abspath(args.project_path)
    summary_file = os.path.join(project_path, args.summary)
    
    print(f"\nProject Directory Structure for: {project_path}\n")
    print_tree(project_path)
    print_summary(summary_file)

if __name__ == "__main__":
    main()
