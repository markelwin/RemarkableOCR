
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


class utils(object):

    @staticmethod
    def normalize_utf8_to_ascii(text):
        """substitutes Latin encoded characters with their ASCII similar equivalent"""
        # NOTE: unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii") # removes too many, ie quotes
        for _hex, _char in LATIN_1_CHARS: text = text.replace(_hex, _char)
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
