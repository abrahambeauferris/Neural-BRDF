import os
import re

def parse_and_update_xyz_material(file_path, output_path):
    """Parses a nano_xyz file, updates the tristimulus blocks, and saves an updated version."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    updated_lines = []
    inside_data_block = False  # To track if we're inside a data block
    last_tristimulus = None  # To track the most recent Tristimulus (X, Y, Z)

    for i, line in enumerate(lines):
        line = line.rstrip()
        
        # Rename Tristimulus to match exemplar (X, Y, Z)
        if re.match(r'TristimulusValue\s+TrisX', line):
            line = 'TristimulusX'
        elif re.match(r'TristimulusValue\s+TrisY', line):
            line = 'TristimulusY'
        elif re.match(r'TristimulusValue\s+TrisZ', line):
            line = 'TristimulusZ'
        
        # If we encounter a new Tristimulus (X, Y, Z), handle the previous block
        if re.match(r'Tristimulus[XYZ]', line):
            # If we were inside a previous data block, we need to close it first
            if inside_data_block and last_tristimulus != line:
                updated_lines.append('DataEnd\n')  # Close previous block with DataEnd
            
            # Append the current Tristimulus (X, Y, Z)
            updated_lines.append(line + '\n')

            # Start the new data block with DataBegin if it isn't already inside a block
            if not inside_data_block:
                updated_lines.append('DataBegin\n')
                inside_data_block = True

            last_tristimulus = line
            continue  # Skip the line to avoid adding it again

        # If DataEnd is explicitly found, we close the current data block
        elif 'DataEnd' in line:
            if inside_data_block:  # Prevent adding a duplicate DataEnd
                updated_lines.append('DataEnd\n')
            inside_data_block = False
            last_tristimulus = None
            continue  # Skip the line

        # If inside data block, handle empty lines as DataEnd marker
        elif inside_data_block and re.match(r'^\s*$', line):
            if not any('DataEnd' in l for l in updated_lines[-3:]):  # Check the last few lines
                updated_lines.append('DataEnd\n')
            inside_data_block = False
            last_tristimulus = None
            continue  # Skip the empty line

        # Otherwise, just add the line
        updated_lines.append(line + '\n')

    # If we finish the file with an open block, we need to close it
    if inside_data_block:
        updated_lines.append('DataEnd\n')

    # Post-process to remove any duplicate DataBegin entries
    post_process_remove_duplicate_data_begin(updated_lines)
    
    with open(output_path, 'w') as output_file:
        output_file.writelines(updated_lines)
    
    print(f"Updated file saved to {output_path}")

def post_process_remove_duplicate_data_begin(lines):
    """Post-process pass to remove any extra DataBegin\nDataBegin occurrences."""
    # Join the lines into a single string, then replace 'DataBegin\nDataBegin' with 'DataBegin'
    file_content = ''.join(lines)
    file_content = file_content.replace('DataBegin\nDataBegin', 'DataBegin')
    
    # Now split the content back into lines
    lines.clear()
    lines.extend(file_content.splitlines(True))

def process_folder(folder_path, output_folder):
    """Processes all nano_xyz files in a folder and saves updated versions to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.brdf'):
            input_path = os.path.join(folder_path, file_name)
            output_path = os.path.join(output_folder, file_name)
            print(f"Processing {input_path}...")
            parse_and_update_xyz_material(input_path, output_path)




if __name__ == '__main__':
    input_folder = '/Users/abraham/Desktop/plasmonicData'
    output_folder = '/Users/abraham/Desktop/ki'  # Directory to save updated files
    process_folder(input_folder, output_folder)


