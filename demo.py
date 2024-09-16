from remarkable import RemarkableOCR, colors
from PIL import Image


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
as_list = list(found.values())  # the start/end only
configs = [dict(highlight_color=colors.starlight),
           dict(highlight_color=colors.green),
           dict(highlight_color=colors.starlight),
           dict(highlight_color=colors.orange, highlight_alpha=0.40),
]

highlight = RemarkableOCR.highlight_statements(im, as_list, data, configs, height_px=height_px)
highlight.show()

# we can redact our secret activities shh :)
phrases = ["I spent the summer reading memos, reports, letters"]
found = RemarkableOCR.find_statements(phrases, data)
as_list = list(found.values())
config = dict(highlight_color=colors.black, highlight_alpha=1.0)
RemarkableOCR.highlight_statements(highlight, as_list, data, config, height_px=height_px).show()
