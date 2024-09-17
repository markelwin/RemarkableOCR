import numpy as np
from PIL import Image

LATIN_1_CHARS = (
    ('\\xe2\\x80\\x99', "'"),
    ('\\xc3\\xa9', 'e'),
    ('\\xe2\\x80\\x90', '-'),
    ('\\xe2\\x80\\x91', '-'),
    ('\\xe2\\x80\\x92', '-'),
    ('\\xe2\\x80\\x93', '-'),
    ('\\xe2\\x80\\x94', '-'),
    ('\\xe2\\x80\\x94', '-'),
    ('\\xe2\\x80\\x98', "'"),
    ('\\xe2\\x80\\x9b', "'"),
    ('\\xe2\\x80\\x9c', '"'),
    ('\\xe2\\x80\\x9c', '"'),
    ('\\xe2\\x80\\x9d', '"'),
    ('\\xe2\\x80\\x9e', '"'),
    ('\\xe2\\x80\\x9f', '"'),
    ('\\xe2\\x80\\xa6', '...'),
    ('\\xe2\\x80\\xb2', "'"),
    ('\\xe2\\x80\\xb3', "'"),
    ('\\xe2\\x80\\xb4', "'"),
    ('\\xe2\\x80\\xb5', "'"),
    ('\\xe2\\x80\\xb6', "'"),
    ('\\xe2\\x80\\xb7', "'"),
    ('\\xe2\\x81\\xba', "+"),
    ('\\xe2\\x81\\xbb', "-"),
    ('\\xe2\\x81\\xbc', "="),
    ('\\xe2\\x81\\xbd', "("),
    ('\\xe2\\x81\\xbe', ")"),
    ('&amp;', 'and'),
)

# a simple map from typographical quotes from newspaper and ocr (e.g. weird ") mapped to their ASCII version
# originally from here; since augmented, simply be careful: https://stackoverflow.com/a/41516221
typographical_unicode = {ord(x): ord(y) for x, y in zip(u"‘’´“”–-—'", u"'''\"\"---'")}




class utils(object):

    @staticmethod
    def normalize_utf8_to_ascii(text):
        """substitutes Latin encoded characters with their ASCII similar equivalent"""
        # NOTE: unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii") # removes too much; quotes
        for _hex, _char in LATIN_1_CHARS: text = text.replace(_hex, _char)
        text = text.translate(typographical_unicode)
        return text

    @staticmethod
    def lowercase_alphanum_normalized(text):
        """simply removes all non-isalnum, makes lowercase, and removes double whitespaces"""
        return " ".join(["".join([c for c in word if c.isalnum()]) for word in text.lower().split()])

    @staticmethod
    def remove_overlapping_duplicates(found, priority=None):
        """removes sequences from found which are subsequences of found or found + priority if priority is not None"""
        # NOTE: sequences are sorted by longest length first; and any overlap disqualifies shorter sequences, i.e.,
        # [4, 7] will disqualify [5, 6] and also [6, 8]. Note that found=Array<[word, start_i, end_i]>
        found.sort(key=lambda x: x[2]-x[1], reverse=True)
        if priority is not None: priority.sort(key=lambda x: x[2]-x[1], reverse=True)
        cleaned = []
        for [w, s, e] in found:
            skip = False
            for (_, sc, ec) in (cleaned if priority is None else priority + cleaned):
                if ec >= s >= sc or ec >= e >= sc:
                    skip = True
                    break
            if not skip: cleaned.append([w, s, e])
        return cleaned

    @staticmethod
    def python_sanitize_numpy(data):
        """Marshals common numpy data package data to native Python types"""
        if isinstance(data, np.ndarray): return data.tolist()
        if type(data).__module__ == np.__name__: return data.item()
        if isinstance(data, (list, tuple)): return [utils.python_sanitize_numpy(item) for item in data]
        if isinstance(data, dict): return {k: utils.python_sanitize_numpy(v) for k,v in data.items()}
        return data


class CONSTANTS(object):
    # typography analysis requires knowledge of which characters extend above the so-called x-height and below so-called
    # baseline. for typography we use these designations: m2.material.io/design/typography/understanding-typography.html
    above_x_height_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZbdfhijklt!@#$%^&()`{}[]\"'?/|\\"
    below_baseline_chars = "gjpqy$;,"
    known_char_lexicon = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrustvwxyz!@#$%^&*()_+=-~`{}[]:;\"'?/><.,|\\"


class plotting(object):

    @staticmethod
    def tile_images(images, n_width=10, tile_wh=None, pad_wh=None):
        """Creates a PIL image with images thumbnailed to tile_wh and tiled n_width in a row"""
        if tile_wh is None: tile_wh = [100, 100]
        if pad_wh is None: pad_wh = [5, 5]

        for i, image in enumerate(images):
            if tile_wh[0] is not None and tile_wh[1] is not None:
                width, height = image.size
                scale_factor = min(tile_wh[0]/width, tile_wh[1]/height)
                resized = image.resize((int(width*scale_factor), int(height*scale_factor)))
                image = Image.new("RGB", tile_wh, color="white")
                image.paste(resized, (0,0))
            elif tile_wh[1] is None and tile_wh[0] is not None:
                image = image.resize((int(tile_wh[0]), int(image.height*tile_wh[0]/image.width)))
            elif tile_wh[0] is None and tile_wh[1] is not None:
                image = image.resize((int(image.width*tile_wh[1]/image.height), int(tile_wh[1])))
            else: raise ValueError("sub-image tile_wh must be [width, height] with one parameter optionally None")
            images[i] = image

        bboxes = []
        for i, image in enumerate(images):
            row, col = int(i/n_width), i%n_width
            top = pad_wh[1] + (0 if row == 0 else bboxes[i-n_width][3])
            left = pad_wh[0] + (0 if col == 0 else bboxes[i-1][2])
            bboxes.append([left, top, left+image.width, top+image.height])

        canvas_wh = max([b[2] for b in bboxes]) + pad_wh[0], max([b[3] for b in bboxes]) + pad_wh[1]
        canvas = Image.new("RGB", canvas_wh, color="white")
        for i, b in enumerate(bboxes): canvas.paste(images[i], box=b)
        return canvas