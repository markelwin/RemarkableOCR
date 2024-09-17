from remarkable import RemarkableOCR, colors, RemarkableOCRResearch, plotting
from PIL import Image
import more_itertools
import random


# Operation Moonglow; annotated by David Bernat
image_filename = "remarkable/_db/docs/moonglow.jpg"
im = Image.open(image_filename)

##################################################################
#  using data
##################################################################
data = RemarkableOCR.ocr(image_filename)

# we can debug using an image
RemarkableOCR.create_debug_image(im, data).show()

# hey. what are all the c words?
cwords = [d for d in data if "sea" in d["text"].lower()]
cwords = RemarkableOCR.create_debug_image(im, cwords).show()

# nevermind; apply filters because this is a book page
# removes annotations on the edges; which are often numerous
data = RemarkableOCR.filter_assumption_blocks_of_text(data)
margins = [d for d in data if d["is_first_in_line"] or d["is_last_in_line"]]
RemarkableOCR.create_debug_image(im, margins).show()

# transforms data to a space-separated string; adding new-lines at paragraph breaks.
readable = RemarkableOCR.readable_lines(data)
print(readable)

##################################################################
#  highlighting
##################################################################

# to create a highlight bar based on token pixel sizes
# if None will calculate on max/min height of the sequence
base = RemarkableOCR.document_statistics(data)
wm, ws = base["char"]["wm"], base["char"]["ws"]
height_px = wm + 6*ws

# simple search for phrases (lowercase, punctuation removed) returns one result for each four
phrases = ["the Space Age", "US Information Agency", "US State Department", "Neil Armstrong"]
found = RemarkableOCR.find_statements(phrases, data)

# we can highlight these using custom highlights
configs = [dict(highlight_color=colors.starlight),
           dict(highlight_color=colors.green),
           dict(highlight_color=colors.starlight),
           dict(highlight_color=colors.orange, highlight_alpha=0.40),
]

highlight = RemarkableOCR.highlight_statements(im, found, data, configs, height_px=height_px)
highlight.show()

# we can redact our secret activities shh :)
phrases = ["I spent the summer reading memos, reports, letters"]
found = RemarkableOCR.find_statements(phrases, data)
config = dict(highlight_color=colors.black, highlight_alpha=1.0)
RemarkableOCR.highlight_statements(highlight, found, data, config, height_px=height_px).show()


##################################################################
#  research & development
##################################################################

# we can use large reoccurrences of words (about ten sentences) to estimate typographical information about individual
# characters, including their typographical baseline and x_height, and typical dimensions of individual characters.
# this statistical procedure is very robust with, and tested with, mostly uniform text fonts (i.e., book pages).
data, typo = RemarkableOCRResearch.enrich_typographical_statistics(data)
if typo is None: raise RuntimeError("typography failed to converge. please contribute this image to an issue")
RemarkableOCRResearch.create_typography_debug_image(im, data).show()

# we can use computer vision to estimate whether images have handwritten underlining; because the typographical features
# provide very helpful constraints on where underlying occurs this feature is only available when typography converges.
data = RemarkableOCRResearch.enrich_handwritten_features(im, data)
hwords = [d for d in data if d["is_highlighted"]]
RemarkableOCR.create_debug_image(im, hwords).show()

# we can also analyze the specific character instances estimated by typographical features. first we show all letters t.
# second we organize the char_bboxes by character and sort by widest character, choosing a random example of each. third
# we have a little fun by generating arbitrary sentences (not recommended for hostage taking or love letters, please).
# this demo uses a utility that takes a list of images and plots them in a tile grid left to right top to bottom.
t_data = [t for word in typo["char_bboxes"] for t in word if t["char"] == "t"]
images = [im.crop(dct["bbox"]) for dct in t_data]
plotting.tile_images(images, tile_wh=[None, 100], n_width=20).show()

char_boxes_by_char = [t for word in typo["char_bboxes"] for t in word]
char_boxes_by_char = more_itertools.map_reduce(char_boxes_by_char, lambda item: item["char"], lambda item: item["bbox"])
chars_by_width = dict(sorted(typo["font_char_widths"].items(), reverse=True, key=lambda item: item[1])).keys()

random.seed(0)
chars_data = [random.choice(char_boxes_by_char[c]) for c in chars_by_width]
images = [im.crop(bbox) for bbox in chars_data]
plotting.tile_images(images, tile_wh=[None, 100], n_width=11).show()

quote = "Same road, no cars. It's magic."
images = []
for word in quote.split(" "):
    chars_data = [random.choice(char_boxes_by_char[c]) for c in word if c != " "]
    as_images = [im.crop(bbox) for bbox in chars_data]
    images.append(plotting.tile_images(as_images, tile_wh=[None, 100], pad_wh=[0,0], n_width=len(word)))
plotting.tile_images(images, tile_wh=[None, 100], pad_wh=[60, 5], n_width=2).show()
