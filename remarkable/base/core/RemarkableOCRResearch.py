from remarkable.base.core.lightweight_utilities.colors import colors
from remarkable.base.core.lightweight_utilities.utils import utils, CONSTANTS
from PIL import ImageDraw
import numpy as np
import skimage
import collections
import heapq

import logging
logger = logging.getLogger(__name__)


class RemarkableOCRResearch(object):

    @staticmethod
    def enrich_typographical_statistics(data):
        """Applies statistics and regression to estimate features such as baseline height and character widths."""

        # We set up a regression to determine the pixel height of three elements: descender to baseline; baseline to x;
        # x to ascender. We can estimate heights in each word by estimating the proportions across all words.
        # NOTE: see 'improvements' in enrich_handwritten_features for a discussion of , and "
        try: 
            A = [[d["amt_below_baseline"], 1, d["amt_above_x_height"]] for d in data]
            Y = [[d["height"]] for d in data]
            results = np.linalg.lstsq(A, Y, rcond=None)
            descender_to_baseline, baseline_to_x_height, x_height_to_ascender = utils.python_sanitize_numpy(results[0].ravel())
        except: 
            logger.error("typography baseline x_height failed to converge. please contribute this image to issues " +\
                         "so we can address: https://github.com/markelwin/RemarkableOCR/issues", exc_info=True)
            return data, None

        # We set up a regression to determine the width of each character; this method allows for characters that are
        # not in known_char_lexicon, but does require using this newly generated char_lexicon for all downstream tasks
        try: 
            char_lexicon = list(set("".join(d["text"] for d in data)))
            A = [[d["text"].count(c) for c in char_lexicon] for d in data]
            Y = [[d["width"]] for d in data]
            results = np.linalg.lstsq(A, Y, rcond=None)
            results = utils.python_sanitize_numpy(results[0].ravel())
            font_char_widths = {char_lexicon[i]:v for i, v in enumerate(results)}
        except:
            logger.error("typography font_char_widths failed to converge. please contribute this image to issues " +\
                         "so we can address: https://github.com/markelwin/RemarkableOCR/issues", exc_info=True)
            return data, None

        for d in data:
            below_baseline = descender_to_baseline * d["amt_below_baseline"]
            above_x_height = x_height_to_ascender * d["amt_above_x_height"]
            below_baseline_fraction = below_baseline / (below_baseline + baseline_to_x_height + above_x_height)
            above_x_height_fraction = above_x_height / (below_baseline + baseline_to_x_height + above_x_height)
            d["font_baseline"] = d["bottom"] - below_baseline_fraction * d["height"]
            d["font_x_height"] = d["top"] + above_x_height_fraction * d["height"]

        # Here we compile an estimation of the bounding boxes of each character in each token, organized by token.
        char_bboxes = []
        for d in data:
            true_total = sum([font_char_widths[c] for c in d["text"]])
            per_char_data = []
            left_px = d["left"]
            for c in d["text"]:
                char_w = font_char_widths[c] / true_total * d["width"]
                c_top = d["top"] if c in CONSTANTS.above_x_height_chars else d["font_x_height"]
                c_bot = d["bottom"] if c in CONSTANTS.below_baseline_chars else d["font_baseline"]
                per_char_data.append(dict(char=c, bbox=[left_px, c_top, left_px+char_w, c_bot]))
                left_px += char_w
            char_bboxes.append(per_char_data)

        typo = dict(descender_to_baseline=descender_to_baseline, baseline_to_x_height=baseline_to_x_height,
                    x_height_to_ascender=x_height_to_ascender, char_bboxes=char_bboxes,
                    font_char_widths=font_char_widths)
        return data, typo

    @staticmethod
    def create_typography_debug_image(im, data):
        """Draws red lines at baseline and x_height of each token to visually confirm typography statistics."""
        _im = im.copy().convert("RGB")
        draw = ImageDraw.Draw(_im)
        for d in data:
            if "font_baseline" not in d or "font_x_height" not in d:
                raise RuntimeError("data missing enrich_typographical_statistics values. did you run that first?")
            draw.line(((d["left"], d["font_baseline"]), (d["right"], d["font_baseline"])), fill=colors.red, width=3)
            draw.line(((d["left"], d["font_x_height"]), (d["right"], d["font_x_height"])), fill=colors.red, width=3)
        return _im

    @staticmethod
    def enrich_handwritten_features(im, data):
        """statistically estimates handwritten features, i.e., annotations of underlining"""

        for d in data:
            if "font_baseline" not in d or "font_x_height" not in d:
                raise RuntimeError("data missing enrich_typographical_statistics values. did you run that first?")

            fraction_above_baseline, fraction_below_baseline = 0.00, 1.0
            baseline_to_x_height = d["font_baseline"] - d["font_x_height"]
            below_word_bbox = (d["left"], d["font_baseline"] - baseline_to_x_height*fraction_above_baseline,
                               d["right"], d["font_baseline"] + baseline_to_x_height*fraction_below_baseline)
            below_word = im.crop(below_word_bbox).convert('L')

            # this algorithm makes a handful of reasonable hand coded assumptions: firstly, we example an area from the
            # word baseline to one unit of baseline to x_height below the baseline. this is an empty space for most
            # words; balancing the confounding problem that for instance a word with a y may be highlighted up near the
            # baseline for most of the word and even strike-through y, which would have been missed using d["bottom"].
            # the next assumption is that all markings darker than 100/255 pixel intensity is simply a pen marking.
            # the algorithm searches this region below the word for the three largest contiguous islands of ink, and
            # sums their pixel counts (n_black3). it is expected that a marking be approximately larger than 20% of
            # the baseline to x_height (i.e., about 1/20 of the bulk height of the word), and filling at least 60% of
            # the horizontal span of the word (to account for partial underlining and weird merging of tokens by -).
            # we find that when this threshold is met the results are robust (about 90% recall and about 90% precision,
            # which is sufficient for extracting somewhat large spans of highlighted text, and certainly enough for the
            # extraction of focal points in most words for machine learning), usually surpassing by a factor of 2 to 3.
            # the errors that this algorithm makes are not sensitive to this factor--errors are from base presumptions
            # known errors:
            # 1. underlines which strike through the word itself (above baseline) or reasonably far below the window
            # 2. underlines which are partial at the end of long word sequences and therefore ambiguous
            # 3. violations of rules above caused by imperfect ocr (e.g. words longer than reality) and therefore fail
            # improvements:
            # 1. the window could extend (only or all the way) to the x_height of the word/line below it, which may be
            #     less than the 1.0 value (dense text) or far more (when the next line ends a paragraph, e.g. up to 2.0)
            # 2. the baseline height is slightly (~3-5%) confounded because , and sometimes " are not precisely the size
            #     of characters that extend to descender or ascender, and hence through off per-token estimates slightly
            is_ink = np.array(below_word) < 100
            contiguous = skimage.measure.label(is_ink)  # returns 2D array where 0 is background & n is the nth island
            area_of_islands = list(collections.Counter(contiguous[contiguous != 0].ravel().tolist()).values())
            expected_area_of_a_highlight = baseline_to_x_height * d["width"] * 0.20 * 0.60
            n_black3 = sum(heapq.nlargest(3, area_of_islands)) / expected_area_of_a_highlight
            d["is_highlighted"] = n_black3 > 1.00
        return data

