from collections import UserDict
from lxml.etree import XMLParser, fromstring
from trafilatura.utils import line_processing, unescape

from mytraf import mydetermine_returnstring


class DictDocument(UserDict):
    """
        dict-like Document class
    """
    __slots__ = [
        'title', 'author', 'url', 'hostname', 'description', 'sitename',
        'date', 'categories', 'tags', 'fingerprint', 'id', 'license',
        'body', 'comments', 'commentsbody', 'raw_text', 'text',
        'language', 'image', 'pagetype'  # 'locale'?
    ]

    def __init__(self, *args, **kwargs):
        super().__init__()
        for slot in self.__slots__:
            setattr(self, slot, None)
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __setitem__(self, key, value):
        if key in self.__slots__:
            setattr(self, key, value)
        self.data[key] = value

    @classmethod
    def load_from_doc(cls, doc):
        '''
            Convert Document() to DictDocument()
        '''
        ndoc = cls()
        for slot in ndoc.__slots__:
            ndoc.__setitem__(slot, getattr(doc, slot))
        return ndoc

    @classmethod
    def load_from_dict(cls, doc_dict):
        '''
            reconstruct the original doc based on the output of `as_dict()`
        '''
        ndoc = xmltodoc(doc_dict.pop('xml_txt'))
        for k, v in doc_dict.items():
            ndoc[k] = v
        return ndoc

    def set_attributes(self, title, author, url, description, site_name, image, pagetype, tags):
        "Helper function to (re-)set a series of attributes."
        if title:
            self.title = title
        if author:
            self.author = author
        if url:
            self.url = url
        if description:
            self.description = description
        if site_name:
            self.sitename = site_name
        if image:
            self.image = image
        if pagetype:
            self.pagetype = pagetype
        if tags:
            self.tags = tags

    def clean_and_trim(self):
        "Limit text length and trim the attributes."
        for slot in self.__slots__:
            value = getattr(self, slot)
            if isinstance(value, str):
                # length
                if len(value) > 10000:
                    new_value = value[:9999] + '…'
                    setattr(self, slot, new_value)
                    value = new_value
                # HTML entities, remove spaces and control characters
                value = line_processing(unescape(value))
                setattr(self, slot, value)

    def as_dict(self, output_format='xml', include_formatting=False, pretty_print=False):
        "Convert the document to a dictionary of string."
        # return {
        #     attr: getattr(self, attr)
        #     for attr in self.__slots__
        #     if hasattr(self, attr)
        # }

        # outputdict = {k:v for k,v in self.data.items()}
        # outputdict['source'] = outputdict.pop('url')
        # outputdict['source-hostname'] = outputdict.pop('sitename')
        # outputdict['excerpt'] = outputdict.pop('description')
        # outputdict['categories'] = ';'.join(outputdict['categories'])
        # outputdict['tags'] = ';'.join(outputdict['tags'])
        # outputdict['text'] = myxmltotxt(outputdict.pop('body'), include_formatting=False, include_images=True)
        # if outputdict['commentsbody'] is not None:
        #     outputdict['comments'] = myxmltotxt(outputdict.pop('commentsbody'), include_formatting=False, include_images=True)
        # else:
        #     del outputdict['commentsbody']
        # return outputdict

        # convert body (commentsbody) and __slots__ to xmlstring
        outputstring, _ = mydetermine_returnstring(self, output_format=output_format,
                                                   include_formatting=include_formatting,
                                                   tei_validation=False, pretty_print=pretty_print)
        # update other items in self.data (but not in __slots__)
        outputdict = {k: v for k, v in self.data.items() if k not in self.__slots__}
        outputdict['xml_txt'] = outputstring
        return outputdict

    def __repr__(self) -> str:
        return self.as_dict().__repr__()

    def __getstate__(self):
        state = self.as_dict()
        return state

    def __setstate__(self, state):
        xml_txt = state.pop('xml_txt')
        parser = XMLParser(remove_blank_text=True)
        output_tree = fromstring(xml_txt, parser)

        # resume body (commentsbody) from xml string
        body = output_tree.xpath('main')[0]
        commentsbody = output_tree.xpath('comments')[0]
        self.body, self.commentsbody = body, commentsbody

        # resume attrib in __slots__ from xml string
        if not hasattr(self, 'data'):
            self.data = dict()
        default_slots = [
            'title', 'author', 'url', 'hostname', 'description', 'sitename',
            'date', 'categories', 'tags', 'fingerprint', 'id', 'license',
            # 'body', 'comments', 'commentsbody', 'raw_text', 'text',
            'language', 'image', 'pagetype'  # 'locale'?
        ]
        attrib_dict = {s: s for s in default_slots}
        attrib_dict['url'] = 'source'
        attrib_dict['description'] = 'excerpt'
        for attrib in attrib_dict:
            self[attrib] = None
            xml_attrib = output_tree.xpath(f'@{attrib_dict[attrib]}')
            if len(xml_attrib) > 0:
                if attrib in ['categories', 'tags']:
                    self[attrib] = xml_attrib[0].split(';')
                else:
                    self[attrib] = xml_attrib[0]

        # resume other attrib in self.data from state
        for k, v in state.items():
            self[k] = v


def xmltodoc(xml_txt):
    '''
        input: plain text in xml format
        output: instance of class `DictDocument()`
        NOTE: the original value of ('comments', 'raw_text', 'image', 'pagetype') cannot be recovered
    '''
    parser = XMLParser(remove_blank_text=True)
    output_tree = fromstring(xml_txt, parser)
    body = output_tree.xpath('main')[0]
    commentsbody = output_tree.xpath('comments')[0]

    document = DictDocument()
    document.body, document.commentsbody = body, commentsbody

    default_slots = [
        'title', 'author', 'url', 'hostname', 'description', 'sitename',
        'date', 'categories', 'tags', 'fingerprint', 'id', 'license',
        # 'body', 'comments', 'commentsbody', 'raw_text', 'text',
        'language', 'image', 'pagetype'  # 'locale'?
    ]
    attrib_dict = {s: s for s in default_slots}
    attrib_dict['url'] = 'source'
    attrib_dict['description'] = 'excerpt'

    for attrib in attrib_dict:
        document[attrib] = None
        xml_attrib = output_tree.xpath(f'@{attrib_dict[attrib]}')
        if len(xml_attrib) > 0:
            if attrib in ['categories', 'tags']:
                document[attrib] = xml_attrib[0].split(';')
            else:
                document[attrib] = xml_attrib[0]

    return document
