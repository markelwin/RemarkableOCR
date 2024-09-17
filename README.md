## RemarkableOCR is a simple ocr tool with improved data, analytics, and rendering tools.

RemarkableOCR creates Image-to-Text positional data and analytics for natural language processing on images. 
RemarkableOCR is based on the Google pytesseract package with additional lightweight processing to make
its **more user-friendly and expansive data**, plus provides one-line simple tools for:
- especially **books**, newspapers, screenshots
- images to **debug**
- **highlights** and **in-doc search**
- **typographical** analysis and **hand-written** annotations.
- and **redaction**.

### installation
```
pip install RemarkableOCR
```


### five-minute demo: data, debug

![demo.data.png](remarkable%2F_db%2Fdocs%2Fdemo.data.png)

```python
from remarkable import RemarkableOCR
from PIL import Image

# Operation Moonglow; annotated by David Bernat
image_filename = "_db/docs/moonglow.jpg"
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
```


### five-minute demo: highlighting

![demo.highlighting.jpg](remarkable%2F_db%2Fdocs%2Fdemo.highlighting.jpg)

```python
from remarkable import RemarkableOCR, colors
from PIL import Image

# Operation Moonglow; annotated by David Bernat
image_filename = "_db/docs/moonglow.jpg"
im = Image.open(image_filename)

##################################################################
#  using data
##################################################################
data = RemarkableOCR.ocr(image_filename)
data = RemarkableOCR.filter_assumption_blocks_of_text(data)

# to create a highlight bar based on token pixel sizes
# if None will calculate on max/min height of the sequence
base = RemarkableOCR.document_statistics(data)
wm, ws = base["char"]["wm"], base["char"]["ws"]
height_px = wm + 6 * ws

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
```

### what is all this data? 

| key  | value      | ours | r&d | description                                                                          |
|:-----|:-----------|:-----|:---|:-------------------------------------------------------------------------------------|
|text| US         |      | | the token text, whitespace removed                                                   |
|conf| 0.96541046 |      |  | confidence score 0 to 1; 0.40 and up is reliable                                     |
|page_num| 1          |      |  | page number will always be 1 using single images                                     |
|block_num| 13         |      |  | a page consists of blocks top to bottom, 1 at top                                    |
|par_num| 1          |      |  | a block consists of paragraphs top to bottom, 1 at top of block                      |
|line_num| 3          |      |  | a paragraph consists of lines top to bottom, 1 at top of paragraph                   |
|word_num| 6          |      |  | a line consists of words left to right, 1 at the far left                            |
|absolute_line_number| 26         | *    |  | line number relative to page as a whole                                              |
|is_first_in_line| False      | *    |  | is the token the left-most in the line?                                              |
|is_last_in_line| False      | *    |  | is the token the right-most in the line?                                             |
|is_punct| False      | *    |  | is every character a punctuation character?                                          |
|is_alnum| True       | *    |  | is every character alphanumeric?                                                     |
|left| 1160.0     |      |  | left-edge pixel value of token bounding box                                          | 
|right| 1238.0     | *    |  | right-edge pixel value of token bounding box                                         |
|top| 2590.0     |      |  | top-edge pixel value of token bounding box                                           |
|bottom| 2638.0     | *    |  | bottom-edge pixel value of token bounding box                                        |
|width| 78.0       |      |  | width pixel value of token bounding box, equal to right minus left                   |
|height| 48.0       |      |  | height pixel value of token bounding box; equal to bottom minus top                  |
|font_size_pt|36.0| * | | simple approximation of font size in pts using 16px = 12pt standard from height      |
|amt_above_x_height| 1.0        |* |* | does character font typically extend above typographical x_height (yes=1.0, no=0.0)  |
|amt_below_baseline| 0.0        |* |* | does character font typically extend below typographical baseline (yes=1.0, no=0.0)  |
|is_highlighted|True|* |* | statistical estimation as to whether the word is underlined by ink or otherwise|
|has_unknown_char|False|*| | whether token contains a character not in our preassigned typography lists           |
|block_left| 116.0      | *    |  | left-edge of block of token; useful for fixed-width cross-line highlighting          |
|block_right| 2195.0     | *    |  | right-edge of block of token; useful for fixed-width cross-line highlighting         |
|level| 5          |      |  | describes granularity of the token, and will always be 5, indicating a token         |

## RemarkableOCR methods to notice

```python
from remarkable import RemarkableOCR
from PIL import Image

filename = "_db/docs/moonglow.jpg"
data = RemarkableOCR.ocr(filename,
                         confidence_threshold=0.50)  # The core RemarkableOCR functionality returns a dictionary of data about each token detected in the image.
data = RemarkableOCR.filter_assumption_blocks_of_text(data,
                                                      confidence_threshold=0.40)  # a filter for identifying one solid block of text; like a book page or newspaper without ads in between
readable = RemarkableOCR.readable_lines(
    data)  # Convenience function to string sequential words to each line; with new lines at breaks; i.e. readable text
stats = RemarkableOCR.document_statistics(
    data)  # Calculate basic statistics of the document itself; i.e., statistics on the pixel size of the font

im = Image.open(filename)
statements = ["Neil Armstrong"]
debug_im = RemarkableOCR.create_debug_image(im,
                                            data)  # Draws a black bounding box around each token to visually confirm every token was identified correctly.
found = RemarkableOCR.find_statements(statements,
                                      data)  # Uses simple regex to identify exact string matches in sequences of tokens, after string normalization
highlight_im = RemarkableOCR.highlight_statements(im, found, data, config=None,
                                                  height_px=None)  # Convenience function for highlighting multiple sequences found=Array<[_, start_i, end_i]> using custom config.
```

### five-minute demo: research features
These are collections of features and improvements which are not thoroughly tested beyond their narrow demonstration
scope, usually books or newspapers; results should be expected to be unstable for numerous edge cases and these APIs
should be considered moderately unstable, but are also most reactive to user feedback. 

![demo.typographics.png](remarkable%2F_db%2Fdocs%2Fdemo.typographics.png)
```python
from remarkable import RemarkableOCR, RemarkableOCRResearch, plotting
from PIL import Image
import more_itertools
import random

# Operation Moonglow; annotated by David Bernat
image_filename = "_db/docs/moonglow.jpg"
im = Image.open(image_filename)
data = RemarkableOCR.ocr(image_filename)
data = RemarkableOCR.filter_assumption_blocks_of_text(data)

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
```


### Licensing & Stuff
<div>
<img align="left" width="100" height="100" style="margin-right: 10px" src="remarkable/_db/docs/starlight.logo.icon.improved.png">
Hey. I took time to build this. There are a lot of pain points that I solved for you, and a lot of afternoons staring 
outside the coffeeshop window at the sunshine. Not years, because I am a very skilled, competent software engineer. But
enough, okay? Use this package. Ask for improvements. Integrate this into your products. Complain when it breaks. 
Reference the package by company and name. Starlight Remarkable and RemarkableOCR. Email us to let us know!
</div>


<br /><br /><br />
Starlight LLC <br />
Copyright 2024 <br /> 
All Rights Reserved <br />
GNU GENERAL PUBLIC LICENSE <br />
