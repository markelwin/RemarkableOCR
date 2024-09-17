from remarkable.base.core.lightweight_utilities.utils import utils, CONSTANTS
from remarkable.base.core.lightweight_utilities.colors import ColorUtils, colors
from PIL import Image, ImageDraw
import statistics
import pytesseract
import tempfile
import string
import re

import logging
logger = logging.getLogger(__name__)


class RemarkableOCR(object):
    """RemarkableOCR creates Image-to-Text positional data and analytics for natural langauge processing on images."""

    @staticmethod
    def ocr(filename, confidence_threshold=0.50):
        """The core RemarkableOCR functionality returns a dictionary of data about each token detected in the image."""
        # text: the text of the token, normalized to remove whitespace and non-ascii characters
        # conf: the confidence score of the detection valued between 0 and 1
        # left, top, right, bottom: the left, top, right, bottom image pixel of the token bounding box, respectively
        # width, height: the width, height of the token bounding box, for convenience, respectively
        # page_num, block_num, par_num, line_num, word_num: pages contain blocks; blocks paragraphs; paragraphs lines.

        # by default we transform to png to avoid various read errors that are inherent in other formats.
        # for example: https://github.com/lovell/sharp/issues/3825#issuecomment-1762708294
        im = Image.open(filename)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=True)  # BytesIO sometimes fail; so use filename
        im.save(tmp.name, format="png")
        data = pytesseract.image_to_data(Image.open(tmp.name))  # if this fails, we throw

        # There is only one data value that breaks convention with pytesseract which is our use of confidence as a
        # number between 0 and 1 instead of 0 and 100. We require enrich_advanced_metadata to not alter the data
        # in a way in which successful calls cannot be made (i.e., user may filter tokens and recalculate safely).
        # For that reason we keep separate the intrinsic per-token adjustments from the enrich_advanced_metadata.
        data = [d.split("\t") for d in data.split("\n")]
        data = [{data[0][k]:data[i][k] for k in range(len(data[0]))} for i in range(1,len(data)-1)]
        data = [d for d in data if len(d["text"].strip()) != 0]
        for d in data:
            d["text"] = utils.normalize_utf8_to_ascii(d["text"])
            d["conf"], d["text"] = float(d["conf"]) / 100., d["text"].strip()
            d["left"], d["top"] = float(d["left"]), float(d["top"])
            d["width"], d["height"] = float(d["width"]), float(d["height"])
            d["right"], d["bottom"] = d["left"] + d["width"], d["top"] + d["height"]
            d["line_num"], d["word_num"] = int(d["line_num"]), int(d["word_num"])
            d["par_num"], d["block_num"], d["page_num"] = int(d["par_num"]), int(d["block_num"]), int(d["page_num"])
        # calculate advanced metadata and return to user
        return RemarkableOCR.enrich_advanced_metadata(data, confidence_threshold)

    @staticmethod
    def enrich_advanced_metadata(data, confidence_threshold=None):
        """Improves per-token data for analytics and rendering; distinguishing blocks; filtering low quality"""
        # is_first_in_line, is_last_in_line: boolean, is the token first or last in the line
        # block_left, block_right: pixel leftmost/rightmost edge of block of token; i.e., highlighting an entire block
        # is_punct: are all characters of text punctuation?
        # is_alnum: are all characters of text alphanumeric?
        # absolute_line_number: line_num refers to line_num within paragraph; absolute_line_number refers to document
        # NOTE: data can be filtered by any external method; and this function recalled to update these values above.

        if confidence_threshold is not None:
            data = [d for d in data if float(d["conf"]) >= confidence_threshold and len(d["text"].strip()) != 0]

        # a block contains paragraphs; paragraphs contain lines; absolute_line_num is the cumulative line in document
        absolute_line_number = 1
        current_block, current_para, current_line = 1, 1, 1
        for d_i, d in enumerate(data):
            # now update current indexes; notice line_num has to be first to check for current_block changes.
            d["is_first_in_line"], d["is_last_in_line"] = d_i == 0, d_i == len(data)-1  # updated below as well
            if d["block_num"] != current_block or d["par_num"] != current_para or d["line_num"] != current_line:
                absolute_line_number += 1
                current_line = d["line_num"]
                d["is_first_in_line"] = True
                if d_i != 0: data[d_i-1]["is_last_in_line"] = True
            if d["block_num"] != current_block or d["par_num"] != current_para:
                current_para = d["par_num"]
            if d["block_num"] != current_block or d_i == len(data)-1:
                block_left, block_right = 1e6, -1e6
                for i in range(d_i+1):
                    if data[i]["block_num"] == current_block:
                        if data[i]["is_first_in_line"]: block_left = min(data[i]["left"], block_left)
                        if data[i]["is_last_in_line"]: block_right = max(data[i]["right"], block_right)
                for i in range(d_i+1):
                    if data[i]["block_num"] == current_block:
                        data[i]["block_left"] = block_left
                        data[i]["block_right"] = block_right
                current_block = d["block_num"]
            d["absolute_line_number"] = absolute_line_number

            char_lexicon = list(set("".join(d["text"] for d in data)))
            char_missing = [c for c in char_lexicon if c not in CONSTANTS.known_char_lexicon]
            if len(char_missing) != 0:
                logger.info(f"NOTE: unknown characters detected in text. slightly incorrect typography statistics " +\
                            f"may result. please report the characters={char_missing} in an issue here and we will " +\
                            f"address this promptly: https://github.com/markelwin/RemarkableOCR/issues")
            d["is_punct"] = all([c in string.punctuation for c in d["text"]])
            d["is_alnum"] = d["text"].isalnum()
            d["has_unknown_char"] = len(char_missing) == 0
            d["amt_below_baseline"] = 1.0 if any([c in CONSTANTS.below_baseline_chars for c in d["text"]]) else 0.0
            d["amt_above_x_height"] = 1.0 if any([c in CONSTANTS.above_x_height_chars for c in d["text"]]) else 0.0
            d["font_size_pt"] = d["height"] / 16.0 * 12.0  # see note in README.md for its description
        return data

    @staticmethod
    def filter_assumption_blocks_of_text(data, confidence_threshold=0.40):
        """a filter for identifying one solid block of text; like a book page or newspaper without ads in between"""
        # NOTE: for right now all this does is remove text to the left and right of the block; i.e., pen annocations
        # NOTE: top/bottom filtering are not necessary now to solve block_left and block_right problems of highlighter.
        # NOTE: always filter by roughly 10%-90% quartiles to get rid of outliers; which bias strongly.

        # get base statistics on the size of the font
        base = RemarkableOCR.document_statistics(data)
        wm, ws = base["char"]["wm"], base["char"]["ws"]

        # using quantiles instead of mean/std is necessary because of variation from partial lines (i.e., end paragraph)
        raw_is_left = [d for d in data if d["is_first_in_line"]]
        raw_left_edge = statistics.quantiles([d["left"] for d in raw_is_left], n=10)[5]
        is_left = [d for d in raw_is_left if d["left"] > raw_left_edge-2.5*wm]
        left_edge = statistics.quantiles([d["left"] for d in is_left], n=10)[0]

        raw_is_right = [d for d in data if d["is_last_in_line"]]
        raw_right_edge = statistics.quantiles([d["right"] for d in raw_is_right], n=10)[-5]
        is_right = [d for d in raw_is_right if d["right"] < raw_right_edge+2.5*wm]
        right_edge = statistics.quantiles([d["right"] for d in is_right], n=10)[-1]

        # The top/bottom filtering are not necessary now to solve block_left and block_right problems of highlighter. It
        # is sufficient to note difficulties: simplest base filter is to search through the top and bottom n (about 5)
        # lines and look for ones that have a full width of pixels (which will make mistakes when first line finished a
        # paragraph from earlier page or paragraph indentation) or to simply iterate through lines where text size and
        # location does not match an anticipated location and statistics for majority of lines; this is the best method.

        data = RemarkableOCR.enrich_advanced_metadata(data, confidence_threshold=confidence_threshold)
        # This fails if multiple tokens on left or right need to be removed: needs to go iteratively through each line.

        # filter left
        to_remove_left = [d for d in raw_is_left if d not in is_left]
        to_remove_right = [d for d in raw_is_right if d not in is_right]
        data = [d for d in data if d not in to_remove_left and d not in to_remove_right and not d["is_punct"]]
        data = RemarkableOCR.enrich_advanced_metadata(data)
        return data

    @staticmethod
    def readable_lines(data):
        """Convenience function to string sequential words to each line; with new lines at breaks; i.e. readable text"""
        if len(data) == 0: return ""
        s = ""
        c_group = [data[0]["page_num"], data[0]["block_num"], data[0]["par_num"], data[0]["line_num"]]
        for d in data:
            group = [d["page_num"], d["block_num"], d["par_num"], d["line_num"]]
            if c_group != group:
                c_group = group
                s += "\n"
            else: s += " "
            s += d["text"]
        s = s.strip()
        return s

    @staticmethod
    def document_statistics(data):
        """Calculate basic statistics of the document itself; i.e., statistics on the pixel size of the font"""
        # char.wm, char.ws, char.hm, char.hs: mean/std for char pixel width, height respectively, filtering outliers
        widths = [(d["right"]-d["left"])/len(d["text"]) for d in data]
        qw = statistics.quantiles(widths, n=10)
        widths = [w for w in widths if qw[-1] > w > qw[0]]
        wm, ws = statistics.mean(widths), statistics.stdev(widths)
        heights = [d["bottom"]-d["top"] for d in data]
        qh = statistics.quantiles(heights, n=10)
        heights = [h for h in heights if qh[-1] > h > qh[0]]
        hm, hs = statistics.mean(heights), statistics.stdev(heights)
        return dict(char=dict(wm=wm, ws=ws, hm=hm, hs=hs))

    @staticmethod
    def uniform_highlight_height_px(data, stddev_factor=6):
        """For instances in which many highlights are made using a statistical height value is aesthetically pleasing"""
        base = RemarkableOCR.document_statistics(data)
        wm, ws = base["char"]["wm"], base["char"]["ws"]
        return wm + stddev_factor*ws

    @staticmethod
    def create_debug_image(im, data):
        """Draws a black bounding box around each token to visually confirm every token was identified correctly."""
        bboxes = [RenderOcr.define_bounding_box_for_tokens(i, i+1, data)[1] for i in range(len(data))]
        bboxes = [x for xs in bboxes for x in xs]
        return RenderOcr.draw_bounding_boxes_for_tokens(im, boundary_lines=bboxes, line_color="#000000")

    @staticmethod
    def highlight_statements(im, found, data, config=None, height_px=None):
        "Convenience function for highlighting multiple sequences found=Array<[_, start_i, end_i]> using custom config."
        for statement_i, (_, start_i, end_i) in enumerate(found):
            highlights, _, _ = RenderOcr.define_bounding_box_for_tokens(start_i, end_i, data, height_px=height_px)
            if config is None: kwargs = dict()
            else: kwargs = config[statement_i] if isinstance(config, (list, tuple)) else config
            im = RenderOcr.draw_bounding_boxes_for_tokens(im, highlights, **kwargs)
        return im

    # METHODS borrowed from form-function specific classes below

    @staticmethod
    def find_statements(statements, data): return FindOcr.find_statements(statements, data)


class RenderOcr(object):
    """Utilities for rendering information from RemarkableOCR onto an image."""

    @staticmethod
    def define_bounding_box_for_tokens(start_i, end_i, data, use_narrowest=True, height_px=None):
        """Construct bbox, underline, and highlight data for a sequence of tokens of high confidence data"""
        # These methods presume very high rates of accurate OCR, and block formatting.
        # such as those expected from OCR of screenshots of web article with no ads and only text.
        # use_narrowest: uses slightly narrower boxes when highlighting across multiple sized tokens
        # height_px: enforces uniform height of boxes across all pixels for a more uniform appearance

        # Two style of bounding boxes are created. The first (bounding_box_lines) consists of rough outlines of what
        # encapsulates each sequence; the second (highlight_lines) consists of what are boxes for highlighter marker.
        # The method also return underlines which are simple a bottom line under the continuous token sequence

        highlight_bboxes = []
        line_start_i = min([data[i]["absolute_line_number"] for i in range(start_i, end_i)])
        line_end_i = max([data[i]["absolute_line_number"] for i in range(start_i, end_i)])+1

        text = " ".join([d["text"] for d in data[start_i:end_i]])
        logger.debug(f"bounding boxes line_start_i={line_start_i} line_end_i={line_end_i} text={text}")

        pad_px, to_block_edge_if_within_px = 3, 50
        for line_i in range(line_start_i, line_end_i):
            line_token_start_i = min([i for i in range(start_i, end_i) if data[i]["absolute_line_number"]==line_i])
            line_token_end_i = max([i for i in range(start_i, end_i) if data[i]["absolute_line_number"]==line_i])+1

            # left, right, top, bottom correspond to the bounding box of the text in /this/ line_i.
            left = min([data[i]["left"] for i in range(line_token_start_i, line_token_end_i)])
            right = max([data[i]["right"] for i in range(line_token_start_i, line_token_end_i)])
            top = [data[i]["top"] for i in range(line_token_start_i, line_token_end_i)]
            top = statistics.quantiles(top, n=5)[0] if use_narrowest and len(top) > 1 else min(top)
            bottom = [data[i]["bottom"] for i in range(line_token_start_i, line_token_end_i)]
            bottom = statistics.quantiles(bottom, n=5)[-1] if use_narrowest and len(bottom) > 1 else max(bottom)
            # block_left, block_right correspond to the same using the bounding of the entire block the line is in.
            # A block is not quite a paragraph, and sentences can span over multiple blocks, depending on typesetting.
            block_left = min([data[i]["block_left"] for i in range(line_token_start_i, line_token_end_i)])
            block_right = max([data[i]["block_right"] for i in range(line_token_start_i, line_token_end_i)])

            # adjusts the height_px is set; i.e., to enforce highlighting consistency across examples
            if height_px is not None: top, bottom = (bottom + top - height_px)/2, (bottom + top + height_px)/2

            # Constructing the boxes is slightly more complex: to make this look nicer, if the first or last token
            # is presumably intended to type-set extend to the end, but does not, because the chuck kerning is not
            # justified spacing, then we push the left or right edge of the box to the block outside as if it were.
            # Then to draw the box around multi-line sentences we are mindful that each line does not need a top
            # or bottom when the span overlaps with the same horizontal pixels spacing of the line above or below.
            if (data[line_token_start_i]["is_first_in_line"] and block_left > left-to_block_edge_if_within_px) or \
                    (line_i != line_start_i): left = block_left  # because for lines that aren't first or last do block
            if (data[line_token_end_i-1]["is_last_in_line"] and block_right < right+to_block_edge_if_within_px) or \
                    (line_i != line_end_i-1): right = block_right
            # if the first or last token is the first or last in the line, and "close" to the block, round up/down

            # to construct the highlight bounding box: we simply outline each span line by line with a small padding.
            # to construct the disjointed bounding box: we use the same data and merge multi-line boxes below
            highlight_bboxes.append([max(0, left-pad_px), max(0, top-pad_px), right+pad_px, bottom+pad_px])
            # constructing the bounds of the border so that the box can be constructed below; add pad_px later.
            # box_lines.append([left, top, right, bottom])  # all four-sets are left, top, right, bottom

        # NOTE: This for loop is the code which handles the multi-line boxes which do or do not need a top/bottom line.
        # In other words, to highlight the text simply requires the padded bounding box of each line, line by line. But
        # to put a convex box around the text requires connecting the line by line bounding boxes that are external to
        # the entire text; i.e., two lines of text directly on top of one another would not want a 'bounding box' line
        # on the bottom of the top line and the top of the bottom line; this would traditionally not be drawn. This
        # actually is harder than it appears because lines may overlap for parts of the line, and not all, done below.
        # these disjointed bboxes are stored as a long list of (x1, y1, x2, y2) line segments encompassing all the text.
        # The underlines are the lines directly under every line of text, line-by-line, as one would underline it all.
        boundary_lines, underlines = [], []
        for i, high_bbox in enumerate(highlight_bboxes):
            boundary_lines.append([high_bbox[0], high_bbox[1], high_bbox[0], high_bbox[3]])  # left edge
            boundary_lines.append([high_bbox[2], high_bbox[1], high_bbox[2], high_bbox[3]])  # right edge
            if i == 0:
                boundary_lines.append([high_bbox[0], high_bbox[1], high_bbox[2], high_bbox[1]])  # top edge, full
            else:
                left_top_inner = max([high_bbox[0], highlight_bboxes[i-1][0]])
                boundary_lines.append([high_bbox[0], high_bbox[1], left_top_inner, high_bbox[1]])  # top edge, left
                right_top_inner = min([high_bbox[2], highlight_bboxes[i-1][2]])
                boundary_lines.append([right_top_inner, high_bbox[1], high_bbox[2], high_bbox[1]])  # top edge, left
            _bottom_full = [high_bbox[0], high_bbox[3], high_bbox[2], high_bbox[3]]  # bottom edge, full
            underlines.append(_bottom_full)
            if i == len(highlight_bboxes)-1: boundary_lines.append(_bottom_full),
            else:
                left_top_inner = max([high_bbox[0], highlight_bboxes[i+1][0]])
                boundary_lines.append([high_bbox[0], high_bbox[3], left_top_inner, high_bbox[3]])  # bottom edge, left
                right_top_inner = min([high_bbox[2], highlight_bboxes[i+1][2]])
                boundary_lines.append([right_top_inner, high_bbox[3], high_bbox[2], high_bbox[3]])  # bottom edge, left
        return highlight_bboxes, boundary_lines, underlines

    @staticmethod
    def draw_bounding_boxes_for_tokens(im, highlight_bboxes=None, boundary_lines=None, underlines=None,
                                       highlight_color=colors.starlight, highlight_alpha=.25,
                                       line_color=colors.starlight, line_w=5):
        """Renders highlights, bboxes, underlines from define_bounding_box_for_tokens in user defined color and shape"""
        highlight_color_rgba = ColorUtils.as_rgb(highlight_color, highlight_alpha)
        line_color_rgb = ColorUtils.as_rgb(line_color)

        _im = im.copy().convert("RGB")
        draw = ImageDraw.Draw(_im, "RGBA")
        if boundary_lines is not None:
            for line in boundary_lines: draw.line(tuple(line), fill=line_color_rgb, width=line_w)
        if underlines is not None:
            for line in underlines: draw.line(tuple(line), fill=line_color_rgb, width=line_w)
        if highlight_bboxes is not None:
            for bbox in highlight_bboxes: draw.rectangle(tuple(bbox), fill=highlight_color_rgba)
        return _im


class FindOcr(object):
    """Functions for searching for text within the data."""

    @staticmethod
    def find_statements(statements, data):
        """Uses simple regex to identify exact string matches in sequences of tokens, after string normalization"""
        text = " ".join(d["text"] for d in data)
        text = utils.lowercase_alphanum_normalized(text)

        found = []
        for s_i, statement in enumerate(statements):
            lower = utils.lowercase_alphanum_normalized(statement)
            if len(lower.strip()) == 0: continue
            for m in re.compile(fr'\b{lower}\b').finditer(text):
                start_i = text[:m.start()].count(" ")
                end_i = text[:m.end()].count(" ") + 1
                found.append([statement, start_i, end_i])
        return found
