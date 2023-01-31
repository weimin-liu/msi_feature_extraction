""""
various patches for mfe"""
import re
import os

import tqdm


class Duplicates:
    """
    when multiple piece of sediments are measured, the coordinates range are the same, so the name needs to be patched
    before processing
    """

    def __init__(self, txt_folder):
        self.order_by_depth = None
        self.txt_folder = txt_folder
        self.txts = [os.path.join(txt_folder, f) for f in os.listdir(txt_folder) if f.endswith('.txt')]

    def find_depth(self):
        """
        find the depth of the sample by the file name
        Returns:

        """
        self.order_by_depth = {}
        for file in self.txts:
            file_name = os.path.basename(file)
            depth = re.findall(r'\d+-\d+', file_name)
            if depth:
                depth = depth[0].split('-')[0]
                self.order_by_depth[depth] = file
        self.order_by_depth = {k: v for k, v in sorted(self.order_by_depth.items(), key=lambda item: int(item[0]))}

    def patch(self):
        """patch the coordinates in the text files"""
        self.find_depth()
        for depth in self.order_by_depth:
            file = self.order_by_depth[depth]
            with open(file, 'r') as f:
                print(f'Patching {file} ...')
                txt_content = f.readlines()
                txt_content_patched = []
                for line in tqdm.tqdm(txt_content):
                    if line.startswith('R00'):
                        line = line.replace('R00', f'R{depth}')
                    txt_content_patched.append(line)
            # save the patched file with a new name, append _patched to the file name
            with open(file.replace('.txt', '_patched.txt'), 'w') as f:
                f.writelines(txt_content_patched)

