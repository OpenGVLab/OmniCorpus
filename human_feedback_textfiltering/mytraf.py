# SIGALRM isn't present on Windows, detect it
try:
    from signal import signal, alarm, SIGALRM
    HAS_SIGNAL = True
except ImportError:
    HAS_SIGNAL = False

import logging
LOGGER = logging.getLogger(__name__)

from trafilatura.core import timeout_handler, content_fingerprint, bare_extraction, strip_double_tags, remove_empty_elements, \
    build_tei_output, txttocsv, build_json_output, normalize_unicode
from trafilatura.settings import use_config, DEFAULT_CONFIG

from myxml import build_xml_output, control_xml_output, xmltotxt

def myextract(filecontent, url=None, record_id=None, no_fallback=False,
            favor_precision=False, favor_recall=False,
            include_comments=True, output_format='txt',
            tei_validation=False, target_language=None,
            include_tables=True, include_images=False, include_formatting=False,
            include_links=False, deduplicate=False,
            date_extraction_params=None,
            only_with_metadata=False, with_metadata=False,
            max_tree_size=None, url_blacklist=None, author_blacklist=None,
            settingsfile=None, config=DEFAULT_CONFIG,
            **kwargs):
    """Main function exposed by the package:
       Wrapper for text extraction and conversion to chosen output format.

    Args:
        filecontent: HTML code as string.
        url: URL of the webpage.
        record_id: Add an ID to the metadata.
        no_fallback: Skip the backup extraction with readability-lxml and justext.
        favor_precision: prefer less text but correct extraction.
        favor_recall: when unsure, prefer more text.
        include_comments: Extract comments along with the main text.
        output_format: Define an output format:
            'txt', 'csv', 'json', 'xml', or 'xmltei'.
        tei_validation: Validate the XML-TEI output with respect to the TEI standard.
        target_language: Define a language to discard invalid documents (ISO 639-1 format).
        include_tables: Take into account information within the HTML <table> element.
        include_images: Take images into account (experimental).
        include_formatting: Keep structural elements related to formatting
            (only valuable if output_format is set to XML).
        include_links: Keep links along with their targets (experimental).
        deduplicate: Remove duplicate segments and documents.
        date_extraction_params: Provide extraction parameters to htmldate as dict().
        only_with_metadata: Only keep documents featuring all essential metadata
            (date, title, url).
        max_tree_size: Discard documents with too many elements.
        url_blacklist: Provide a blacklist of URLs as set() to filter out documents.
        author_blacklist: Provide a blacklist of Author Names as set() to filter out authors.
        settingsfile: Use a configuration file to override the standard settings.
        config: Directly provide a configparser configuration.

    Returns:
        A Python dict() containing all the extracted information or None.

    """
    # older, deprecated functions
    if kwargs and any([
        # output formats
            'csv_output' in kwargs,
            'json_output' in kwargs,
            'tei_output' in kwargs,
            'xml_output' in kwargs
        ]):
        raise NameError(
            'Deprecated argument: use output_format instead, e.g. output_format="xml"'
            )
        # todo: add with_metadata later

    # configuration init
    config = use_config(settingsfile, config)

    allow_signal = HAS_SIGNAL
    # put timeout signal in place
    if HAS_SIGNAL is True:
        timeout = config.getint('DEFAULT', 'EXTRACTION_TIMEOUT')
        if timeout > 0:
            try:
                signal(SIGALRM, timeout_handler)
                alarm(timeout)
            except ValueError:
                allow_signal = False

    # extraction
    try:
        document = bare_extraction(
            filecontent, url=url, no_fallback=no_fallback,
            favor_precision=favor_precision, favor_recall=favor_recall,
            include_comments=include_comments, output_format=output_format,
            target_language=target_language, include_tables=include_tables,
            include_images=include_images,
            include_formatting=include_formatting, include_links=include_links,
            deduplicate=deduplicate,
            date_extraction_params=date_extraction_params,
            only_with_metadata=only_with_metadata, with_metadata=with_metadata,
            max_tree_size=max_tree_size, url_blacklist=url_blacklist,
            author_blacklist=author_blacklist,
            as_dict=False, config=config,
        )
    except RuntimeError:
        LOGGER.error('Processing timeout for %s', url)
        document = None

    # deactivate alarm signal
    if HAS_SIGNAL is True and allow_signal and timeout > 0:
        alarm(0)

    # post-processing
    if document is None:
        return None
    if output_format != 'txt':
        # add record ID to metadata
        document.id = record_id
        # calculate fingerprint
        document.fingerprint = content_fingerprint(str(document.title) + " " + document.raw_text)

    # return
    # return determine_returnstring(document, output_format, include_formatting, tei_validation)
    return document


def mydetermine_returnstring(document, output_format, include_formatting, tei_validation, pretty_print=True):
    '''Convert XML tree to chosen format, clean the result and output it as a string'''
    # assert 'xml' in output_format or 'json' in output_format
    # XML (TEI) steps
    num_images = 0
    if 'xml' in output_format:
        # last cleaning
        for element in document.body.iter('*'):
            if element.tag != 'graphic' and len(element) == 0 and not element.text and not element.tail:
                parent = element.getparent()
                if parent is not None:
                    parent.remove(element)
            elif element.tag == 'graphic':
                # TODO add more strict image check rules
                url = element.get('src', '')
                if not url: url = element.get('data-src', '')
                if url:
                    # print(url)
                    num_images += 1
        # build output trees
        strip_double_tags(document.body)
        remove_empty_elements(document.body)
        if output_format == 'xml':
            output = build_xml_output(document)
        elif output_format == 'xmltei':
            output = build_tei_output(document)
        # can be improved
        returnstring = control_xml_output(output, output_format, tei_validation, document, pretty_print=pretty_print)
    # # CSV
    # elif output_format == 'csv':
    #     posttext = xmltotxt(document.body, include_formatting)
    #     if document.commentsbody is not None:
    #         commentstext = xmltotxt(document.commentsbody, include_formatting)
    #     else:
    #         commentstext = ''
    #     returnstring = txttocsv(posttext, commentstext, document)
    # JSON
    elif output_format == 'json':
        returnstring = build_json_output(document)
    # TXT
    else:
        returnstring = xmltotxt(document.body, include_formatting)
        if document.commentsbody is not None:
            returnstring += '\n' + xmltotxt(document.commentsbody, include_formatting)
            returnstring = returnstring.strip()
    # normalize Unicode format (defaults to NFC)
    return normalize_unicode(returnstring), num_images


