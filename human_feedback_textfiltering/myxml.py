import logging
from lxml.etree import Element, RelaxNG, SubElement, XMLParser, fromstring, tostring
from html import unescape

WITH_ATTRIBUTES = {'cell', 'del', 'graphic', 'head', 'hi', 'item', 'list', 'ref'}

from trafilatura.xml import replace_element_text, NEWLINE_ELEMS, SPECIAL_FORMATTING, CONTROL_PARSER, sanitize, validate_tei

LOGGER = logging.getLogger(__name__)

def clean_attributes(tree):
    '''Remove unnecessary attributes.'''
    for elem in tree.iter('*'):
        if elem.tag not in WITH_ATTRIBUTES:
            elem.attrib.clear()
    return tree

def control_xml_output(output_tree, output_format, tei_validation, docmeta, pretty_print=True):
    '''Make sure the XML output is conform and valid if required'''
    control_string = sanitize(tostring(output_tree, encoding='unicode'))
    # necessary for cleaning
    output_tree = fromstring(control_string, CONTROL_PARSER)
    # validate
    if output_format == 'xmltei' and tei_validation is True:
        result = validate_tei(output_tree)
        LOGGER.debug('TEI validation result: %s %s %s', result, docmeta.id, docmeta.url)
    return tostring(output_tree, pretty_print=pretty_print, encoding='unicode').strip()

def xmltotxt(xmloutput, include_formatting, include_images=True):
    '''Convert to plain text format and optionally preserve formatting as markdown.'''
    returnlist = []
    # strip_tags(xmloutput, 'div', 'main', 'span')
    # iterate and convert to list of strings
    for element in xmloutput.iter('*'):
        if element.text is None and element.tail is None:
            if element.tag == 'graphic' and include_images:
                # add source, default to ''
                text = element.get('title', '')
                if element.get('alt') is not None:
                    text += ' ' + element.get('alt')
                url = element.get('src', '')
                if not url: url = element.get('data-src', '')
                if url: returnlist.extend(['![', text, ']', '(', url, ')'])
            # newlines for textless elements
            if element.tag in ('graphic', 'row', 'table'):
                returnlist.append('\n')
            continue
        # process text
        textelement = replace_element_text(element, include_formatting)
        # common elements
        if element.tag in NEWLINE_ELEMS:
            returnlist.extend(['\n', textelement, '\n'])
        # particular cases
        elif element.tag == 'item':
            returnlist.extend(['\n- ', textelement, '\n'])
        elif element.tag == 'cell':
            returnlist.extend(['|', textelement, '|'])
        elif element.tag == 'comments':
            returnlist.append('\n\n')
        else:
            if element.tag not in SPECIAL_FORMATTING:
                LOGGER.debug('unprocessed element in output: %s', element.tag)
            returnlist.extend([textelement, ' '])
    return unescape(sanitize(''.join(returnlist)))

def build_xml_output(docmeta):
    '''Build XML output tree based on extracted information'''
    output = Element('doc')
    output = add_xml_meta(output, docmeta)
    docmeta.body.tag = 'main'
    # clean XML tree
    output.append(clean_attributes(docmeta.body))
    if docmeta.commentsbody is not None:
        docmeta.commentsbody.tag = 'comments'
        output.append(clean_attributes(docmeta.commentsbody))
# XML invalid characters
# https://chase-seibert.github.io/blog/2011/05/20/stripping-control-characters-in-python.html
    return output

def add_xml_meta(output, docmeta):
    '''Add extracted metadata to the XML output tree'''
    # metadata
    if docmeta:
        if docmeta.sitename is not None:
            output.set('sitename', docmeta.sitename)
        if docmeta.title is not None:
            output.set('title', docmeta.title)
        if docmeta.author is not None:
            output.set('author', docmeta.author)
        if docmeta.date is not None:
            output.set('date', docmeta.date)
        if docmeta.url is not None:
            output.set('source', docmeta.url)
        if docmeta.hostname is not None:
            output.set('hostname', docmeta.hostname)
        if docmeta.description is not None:
            output.set('excerpt', docmeta.description)
        if docmeta.categories is not None:
            try:
                output.set('categories', ';'.join(docmeta.categories))
            except:
                pass
        if docmeta.tags is not None:
            try:
                output.set('tags', ';'.join(docmeta.tags))
            except:
                pass
        if docmeta.license is not None:
            output.set('license', docmeta.license)
        if docmeta.id is not None:
            output.set('id', docmeta.id)
        if docmeta.fingerprint is not None:
            output.set('fingerprint', docmeta.fingerprint)
        if docmeta.language is not None:
            output.set('language', docmeta.language)
    return output
