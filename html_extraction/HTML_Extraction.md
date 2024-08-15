# HTML Extraction Tools in OmniCorpus —— magic-html

The main body extraction toolkit of the data engine for OmniCorpus has been merged into [magic-html](https://github.com/opendatalab/magic-html), which has been significantly improved compared to the commonly used [trafilatura](https://github.com/adbar/trafilatura). 

In terms of accuracy, we have addressed the issue where trafilatura would overlook the main content of an HTML document when extracting images, and enhanced its capability to handle Chinese, Japanese, and Arabic documents. Additionally, we have incorporated techniques to trim web noise regions based on HTML structure (such as clusters of lists and navigation bars) and style (targeting elements like advertisements, comments, JavaScript, and CSS). 

In terms of efficiency, we optimized the process based on HTML nodes and streamlined the processing pipeline by eliminating the fallback process in challenging cases. With these two improvements, we can not only extract more informative content from the main body but also double the speed of the extraction process.

We present a demo to compare [extraction results of trafilatura](../demos/html_extraction_demo/demo_trafilatura_extraction.html) with [ours](../demos/html_extraction_demo/demo_trafilatura_extraction.html). (You can download and then browse with a browser.)



## Features of magic-html

- Flexible export construction. (HTML or customizable TXT/MarkDown)
- Supports extraction of both pure textual and multimodal corpora.
- Robust for various layout. (Such as articles/forums)
- Support Latex formula extraction and transforming.



## Installation and Usage

Install with pip wheel:

```shell
pip install https://github.com/opendatalab/magic-html/releases/download/magic_html-0.1.2-released/magic_html-0.1.2-py3-none-any.whl
```

Extract the main body of a demo HTML:

```python
from magic_html import GeneralExtractor

# initialize the extractor
extractor = GeneralExtractor()

url = "http://example.com/"
html = """

<!doctype html>
<html>
<head>
    <title>Example Domain</title>

    <meta charset="utf-8" />
    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />  
</head>

<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents. You may use this
    domain in literature without prior coordination or asking for permission.</p>
    <p><a href="https://www.iana.org/domains/example">More information...</a></p>
</div>
</body>
</html>
"""

# extract main content of articles HTML
data = extractor.extract(html, base_url=url)

# extract main content of forum HTML
# data = extractor.extract(html, base_url=url, html_type="forum")

# extract main content of WeChat official accounts HTML
# data = extractor.extract(html, base_url=url, html_type="weixin")

print(data)
```



## Others

LICENSE: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)

Acknowledgments: 

- [trafilatura](https://github.com/adbar/trafilatura)
- [readability-lxml](https://github.com/buriy/python-readability)

