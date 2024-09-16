## RemarkableOCR is a simple ocr tool with improved data, analytics, and rendering tools.

RemarkableOCR creates Image-to-Text positional data and analytics for natural language processing on images. 
RemarkableOCR is based on the Google pytesseract package with additional lightweight processing to make
its **more user-friendly and expansive data**, plus provides one-line simple tools for:
- especially **books**, newspapers, screenshots
- images to **debug**
- **highlights** and **in-doc search**
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
```

### what is all this data? 

| key  | value | ours | description                                                                  |
|:-----|:------|:-----|:-----------------------------------------------------------------------------|
|text|US|      | the token text, whitespace removed                                           |
|conf|0.96541046|      | confidence score 0 to 1; 0.40 and up is reliable                             |
|page_num|1|      | page number will always be 1 using single images                             |
|block_num|13|      | a page consists of blocks top to bottom, 1 at top                            |
|par_num|1|      | a block consists of paragraphs top to bottom, 1 at top of block              |
|line_num|3|      | a paragraph consists of lines top to bottom, 1 at top of paragraph           |
|word_num|6|      | a line consists of words left to right, 1 at the far left                    |
|absolute_line_number|26| *    | line number relative to page as a whole                                      |
|is_first_in_line|False| *    | is the token the left-most in the line?                                      |
|is_last_in_line|False| *    | is the token the right-most in the line?                                     |
|is_punct|False| *    | is every character a punctuation character?                                  |
|is_alnum|True| *    | is every character alphanumeric?                                             |
|left|1160.0|      | left-edge pixel value of token bounding box                                  | 
|right|1238.0| *    | right-edge pixel value of token bounding box                                 |
|top|2590.0|      | top-edge pixel value of token bounding box                                   |
|bottom|2638.0| *    | bottom-edge pixel value of token bounding box                                |
|width|78.0|      | width pixel value of token bounding box, equal to right minus left           |
|height|48.0|      | height pixel value of token bounding box; equal to bottom minus top          |
|block_left|116.0| *    | left-edge of block of token; useful for fixed-width cross-line highlighting  |
|block_right|2195.0| *    | right-edge of block of token; useful for fixed-width cross-line highlighting |
|level|5|      | describes granularity of the token, and will always be 5, indicating a token |

### RemarkableOCR methods to notice

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
