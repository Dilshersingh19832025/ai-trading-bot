import os

def display_py_files_in_cmd(directory, max_lines_per_part=50):
    """
    Opens each .py file in the specified directory, prints its name as a heading,
    and displays its contents in the command prompt in parts.

    :param directory: The directory containing the .py files.
    :param max_lines_per_part: Maximum number of lines to display per part.
    """
    # Iterate through all files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Print the file name as a heading
                print("\n" + "=" * 50)
                print(f"File: {file}")
                print("=" * 50 + "\n")
                # Open and print the file contents with UTF-8 encoding
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        part_number = 1
                        for i in range(0, len(lines), max_lines_per_part):
                            print(f"\nPart {part_number}:\n")
                            print("".join(lines[i:i + max_lines_per_part]))
                            print("\n" + "-" * 50 + "\n")
                            part_number += 1
                            input("Press Enter to continue to the next part...")
                except UnicodeDecodeError:
                    print(f"Error: Could not read {file} due to encoding issues.")

# Example usage
directory_path = r"C:\Users\kaur_\OneDrive\Documents\AI project"  # Replace with your project directory
display_py_files_in_cmd(directory_path)