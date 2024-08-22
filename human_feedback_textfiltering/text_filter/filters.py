import re
import string
from lxml import etree
from nltk import word_tokenize
import difflib

from .filtering_utils import compute_flagged_word_ratio, HEADER_FOOTER_KEYWORDS, IMAGE_CAPTION_WORDS, \
    MERRIAM_WEBSTER_DICTIONARY, Is_finished, remove_incomplete_sentence, remove_incomplete_sentence_last
import jieba
import os


def filter_download_links(xml):
    """Filter out paragraphs with specific phrases and length less than 20."""
    modified = False
    phrases_to_exclude = [
        re.compile(r"For details on the", re.IGNORECASE),
        re.compile(r"the link is", re.IGNORECASE),
    ]

    for paragraph in xml.xpath('//p'):
        if paragraph.text is not None:
            original_text = paragraph.text
            for phrase_pattern in phrases_to_exclude:
                match = phrase_pattern.search(original_text)
                if match:
                    paragraph.getparent().remove(paragraph)
                    modified = True
                    break

    return xml, modified


def filter_source_references(xml):
    """Filter out source references from paragraphs."""
    modified = False
    source_patterns = [
        re.compile(r"^(Special thanks)", re.IGNORECASE),
        re.compile(r"Published\s*by", re.IGNORECASE),
        re.compile(r"All\s*photo\s*credits\s*and\s*copyright", re.IGNORECASE),
        re.compile(r"\(cf\..*?\)$"),
        re.compile(r"\([\w\s&,]+\s*,\s*\d{4,}\)$")
    ]

    for paragraph in xml.xpath('//p'):
        if paragraph.text is not None:
            original_text = paragraph.text
            for pattern in source_patterns:
                match = pattern.search(original_text)
                if match:
                    # 检查是否是后两个匹配，并且匹配位置在段落末尾
                    if pattern in source_patterns[-2:] and match.end() == len(original_text):
                        paragraph.getparent().remove(paragraph)
                        modified = True
                    elif pattern not in source_patterns[-2:]:
                        paragraph.getparent().remove(paragraph)
                        modified = True
                    elif pattern == source_patterns[0] and match.start() == 0:
                        paragraph.getparent().remove(paragraph)
                        modified = True
                    break

    return xml, modified


def social_media_discard(xml):
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None

        if text is None:
            continue
        deleted = False
        for tag in ['references', 'like', 'likes', 'share', 'shares', 'comment', 'comments', 'review', 'reviews',
                    'reviews about', 'call us at', 'email us at', 'more images', 'users online', 'copyright',
                    'subscribe', 'subscribes']:
            # if text is like regex pattern: ^.*references *:? *$
            if re.match(f'^.*{tag} *:? *$', text.lower()):
                paragraph.getparent().remove(paragraph)
                modified = True
                deleted = True
                break
        if deleted:
            continue
        for tag in ['for more info', 'for more detail', 'top threads', 'most read', 'most viewed', 'most shared',
                    'most liked', 'more here']:
            if tag in text.lower():
                paragraph.getparent().remove(paragraph)
                modified = True
                deleted = True
                break
    return xml, modified


def uppercase_discard(xml, uppercase_threshold=0.8, immutable=False):  # edited
    """
    1. If it is mainly composed of uppercase characters (discard);
    """
    all_lines = etree.tostring(
        xml, encoding='utf-8').decode('utf-8').split('<')
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None

        if text is None:
            continue
        uppercase_char_count = sum(char.isupper() for char in text)
        total_char_count = sum(char not in string.whitespace for char in text)  # not simply len(text)
        # print('ratio: ', uppercase_char_count / total_char_count)
        if total_char_count > 0 and (uppercase_char_count / total_char_count) > uppercase_threshold:
            # immunize paragraphs with graphic retention
            if immutable:
                try:
                    line_number = paragraph.sourceline
                    # if line_number is None:
                    #     print('NOTE!! line_number is None, this iteration will be continued.')
                    start_line = max(0, line_number - 10)
                    end_line = min(len(all_lines), line_number + 10)
                    content = ''.join(all_lines[start_line:end_line])
                    if 'graphic' not in content:
                        paragraph.getparent().remove(paragraph)
                        modified = True
                except:
                    continue
            else:
                paragraph.getparent().remove(paragraph)
                modified = True
    return xml, modified


def underlines_split(xml, split_threshold=15):  # edited
    '''
    line level
    Problem 10.
    If there are multiple adjacent underlines in particular item, split it.
    '''
    modified = False
    element_to_discard = []
    pattern = f"_{{{split_threshold},}}"

    for element in xml.xpath('//*'):
        if element.text is not None and element.tail is not None:
            text = ''.join([element.text, element.tail])
        elif element.text is not None:
            text = element.text
        elif element.tail is not None:
            text = element.tail
        else:
            text = None

        if text is None:
            continue
        apperance = re.findall(pattern, text)
        if apperance is None or len(apperance) == 0:
            continue
        modified = True
        sequence = re.split(pattern, text)
        sequence = [s for s in sequence if len(s) > 0]
        if len(sequence) > 1:
            modified = True
            element.text = sequence[0]
            element.tail = ""
            for i in range(len(sequence) - 1, 0, -1):
                new_element = etree.Element(element.tag)
                new_element.text = sequence[i]
                element.addnext(new_element)
                element = new_element
        elif len(sequence) == 1:
            element.text = sequence[0]
            sequence = re.split(pattern, element.text)
            sequence = [s for s in sequence if len(s) > 0]
            element.text = sequence[0]
            element.tail = sequence[0]
            sequence = re.split(pattern, element.tail)
            sequence = [s for s in sequence if len(s) > 0]
            element.tail = sequence[0]
        else:
            element_to_discard.append(element)

    for element in element_to_discard:
        element.getparent().remove(element)

    return xml, modified


def tooshort_discard(xml, short_threshold=4):
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None

        if text is None:
            continue
        if len(text.split()) <= short_threshold:
            # print('text: ', text)
            paragraph.getparent().remove(paragraph)
            modified = True
    return xml, modified


def re_search_line(text, pat_list):
    for pat in pat_list:
        if re.search(pat, text) is not None:
            return True


def phonenum_author_time_discard(xml):
    pat_list_phonenum = [
        r'\d{3}-\d{3}-\d{4}',  # phone number
        r'\(\d{3}\)\s?\d{3}-\d{4}',  # phone number
    ]
    pat_list_time = [
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},\s\d{4}',
        # Month DD, YYYY
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2},\s\d{4}',  # Mon DD, YYYY
        r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}',  # DD Mon YYYY
        r'\b\d{1,2}[:.]\d{2}\s*(a|p)?\.?m\.?\b',  # time: 12:30pm, 12:30 pm, 12:30 PM, 12:30PM, 12:30PM, 12.30pm et al.
        r'\b([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?\b',  # time 12:30, 12:30:45, 1:30, 01:30, 01:30:45 et al.
        r'\b([01]?[0-9]|2[0-3])(:[0-5][0-9])?([ap]\.?m\.?)\b',  # 12pm, 12:30pm, 12:30:45pm, 1pm, 01pm, 01:30pm et al.
        r'\b([01]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\.\d{1,6}\b',  # 12:30:45.123, 1:30:45.123, 01:30:45.123456 et al.
    ]
    pat_list_author = [
        r'\(Photo:[A-Z,a-z,\s]{0,20}\)'
    ]

    pat_list_phonenum = [re.compile(pat) for pat in pat_list_phonenum]
    pat_list_time = [re.compile(pat) for pat in pat_list_time]
    pat_list_author = [re.compile(pat) for pat in pat_list_author]
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            continue

        if len(text.split(' ')) < 80 and re_search_line(text, pat_list_phonenum):
            parent = paragraph.getparent()
            if parent is not None:
                parent.remove(paragraph)
            modified = True
            # print(text)
            # print(len(text.split(' ')))
            continue

        if len(text.split(' ')) < 20 and re_search_line(text, pat_list_time):
            parent = paragraph.getparent()
            if parent is not None:
                parent.remove(paragraph)
            modified = True
            # print(text)
            # print(len(text.split(' ')))
            continue

        if len(text.split(' ')) < 10 and (
                text.strip().lower().startswith('by ') or \
                re_search_line(text, pat_list_author)
        ):
            parent = paragraph.getparent()
            if parent is not None:
                parent.remove(paragraph)
            modified = True
            # print(text)
            # print(len(text.split(' ')))
            continue

    for head in xml.xpath('//head'):
        text = head.text

        if text is not None and len(text.split(' ')) < 10 and (
                text.strip().lower().startswith('by ') or \
                re_search_line(text, pat_list_author)
        ):
            parent = head.getparent()
            if parent is not None:
                parent.remove(head)
            modified = True
            # print(text)
            # print(len(text.split(' ')))
            continue

    return xml, modified


def aberrant_item_discard(xml, word_length_thr: int = 2, debug_mode: bool = False):
    modified = False
    xml_lines_before = etree.tostring(xml, pretty_print=True).decode().splitlines()

    for item in xml.xpath('//item'):
        if item.text is not None and item.text.startswith('Things to do'):
            item.getparent().remove(item)

    xml_lines_after = etree.tostring(xml, pretty_print=True).decode().splitlines()
    return xml, modified
    differ = difflib.Differ()

    for line in list(differ.compare(xml_lines_before, xml_lines_after)):
        if line.startswith('+'):
            if debug_mode and not modified: print('\n' + '#' * 30 + ' DIFF ' + '#' * 30)
            if debug_mode: print(f'\033[32m{line}\033[0m')
            modified = True
        elif line.startswith('-'):
            if debug_mode and not modified: print('\n' + '#' * 30 + ' DIFF ' + '#' * 30)
            if debug_mode: print(f'\033[34m{line}\033[0m')
            modified = True
    if debug_mode and modified: print('#' * 30 + ' DIFF ' + '#' * 30 + '\n')
    return xml, modified


def cite_discard(xml, debug_mode: bool = False):
    """
    Cases: 2 -> square_brackets_discard
        <p>[1] A pdf of the book is available, without charge, at https://606c3aae-c155-42c4-8bca-a7e28b237aee.filesusr.com/ugd/a03a76_9762bf3ef02d42f28aa8c9e6362bbc21.pdf .</p>
        <p>[2] David Sciulli, Etzioni&#8217;s Critical Functionalism: Communitarian Origins and Principles , International Studies in Sociology and Social Anthropology (Leiden and Boston: Brill, 2011), vii, Proquest Ebook Central.</p>
        <p>[3] Richard A. Posner, Public Intellectuals: A Study of Decline: With a New Preface and Epilogue (Cambridge, MA and London: Harvard University Press, 2001, 2003), 212&#8211;213, Proquest Ebook Central.</p>
        <p>[4] Ian Buruma, &#8220;Are China and the United States Headed for War?&#8221; New Yorker , June 12, 2017, https://www.newyorker.com/magazine/2017/06/19/are-china-and-the-united-states-headed-for-war</p>
        <p>[5] Download the book without charge at https://link.springer.com/chapter/10.1007%2F978-3-319-69623-2_1 .</p>
    Cases: 6 -> reference_sponsore_discard
        <head rend="h1">References:</head>
        <list rend="ul">
            <item>Chamberlin, Donald (2012). &#8220;Early History of SQL&#8221;. IEEE Annals of the History of Computing . 34 (4): 78&#8211;82. doi : 10.1109/MAHC.2012.61 . S2CID 1322572</item>
            <item>&#8220;Welcome to the Course!: SQL.&#8221; Campus.datacamp.com , campus.datacamp.com/courses/introduction-to-sql/sorting-and-grouping.</item>
        </list>
    Cases: 41 -> time_person_email_phonenum_url_discard
        <p>Follow@htshowbizfor more<lb/>The author tweets @RohanNaahar</p>
    Cases: 46 -> time_person_email_phonenum_url_discard
        <quote>
            <p>The onward march of Medicaid expansion also suggests that people are expecting SCOTUS to laugh away Texas&#8217;s anti-Obamacare lawsuit. https://t.co/HhVZHL6Tfh</p>
            <p>&#8212; Dave Weigel (@daveweigel) January 9, 2020</p>
        </quote>
    Cases: 47 -> reference_sponsore_discard
        <head rend="h2">Reference [ ]</head>
        <p>1 First meets Kevin in the Heart of the Storm</p>
        <p>2 Seen walking down an alleyway.</p>
    Cases: 53 -> reference_sponsore_discard
        <p>Sponsored By</p>
    Cases: 84 -> time_person_email_phonenum_url_discard
        <p>* Loose Women is on weekdays, ITV at 12.30pm</p>
    Cases: 115 -> time_person_email_phonenum_url_discard
        <p>This article was originally published on November 30, 2015.</p>
        <p>Support the news</p>
    Cases: 120 -> time_person_email_phonenum_url_discard
        <p>(Reporting by David Shepardson; Editing by Kenneth Maxwell)</p>
    Cases: 124 -> time_person_email_phonenum_url_discard
        <p>Adapted from this Dan's Papers article .</p>
    Cases: 126 -> last_bracket_parag_discard
        <p>(Except for the headline, this story has not been edited by NDTV staff and is published from a syndicated feed.)</p>
    Cases: 149 -> time_person_email_phonenum_url_discard
        <p>Gretchen McCartney
            <lb/>Jet Propulsion Laboratory, Pasadena, Calif.
            <lb/>818-393-6215<lb/>gretchen.p.mccartney@jpl.nasa.gov
        </p>
        <p>Dwayne Brown / JoAnna Wendel<lb/>NASA Headquarters, Washington
            <lb/>202-358-1726 / 202-358-1003<lb/>dwayne.c.brown@nasa.gov / joanna.r.wendel@nasa.gov
        </p>
    Cases: 154 -> time_person_email_phonenum_url_discard
        <p>This article was originally published by AskMen UK.</p>
    Cases: 165 -> time_person_email_phonenum_url_discard
        <head rend="h2">Coverage underway on Sky Sports Arena from 1pm with the evening session on Sky Sports Action from 7pm</head>
    Cases: 185 -> time_person_email_phonenum_url_discard
        <head rend="h2">Brighton and Hove Albion have upheld the decision to sack Gus Poyet as manager of the club for gross misconduct.</head>
        <p>Last Updated: 16/07/13 5:22pm</p>
    Cases: 197 ->
        <p>Home Scene</p>
        <quote>Although this is an exploration game, I want guides to be readily available in case someone is too stumped on a gameplay element.</quote>
        <p>Orange, CA (PRWEB) July 30, 2012</p>
    """
    modified = False
    xml_lines_before = etree.tostring(xml, pretty_print=True).decode().splitlines()

    xml, _ = square_brackets_discard(xml)
    xml, _ = reference_sponsore_discard(xml)
    xml, _ = time_person_email_phonenum_url_discard(xml)
    xml, _ = last_bracket_parag_discard(xml)

    xml_lines_after = etree.tostring(xml, pretty_print=True).decode().splitlines()
    return xml, modified
    differ = difflib.Differ()

    for line in list(differ.compare(xml_lines_before, xml_lines_after)):
        if line.startswith('+'):
            if debug_mode and not modified: print('\n' + '#' * 30 + ' DIFF ' + '#' * 30)
            if debug_mode: print(f'\033[32m{line}\033[0m')
            modified = True
        elif line.startswith('-'):
            if debug_mode and not modified: print('\n' + '#' * 30 + ' DIFF ' + '#' * 30)
            if debug_mode: print(f'\033[34m{line}\033[0m')
            modified = True
    if debug_mode and modified: print('#' * 30 + ' DIFF ' + '#' * 30 + '\n')
    return xml, modified


def last_bracket_parag_discard(xml):
    # 去掉最后一个 p 是被包在括号里的，基本都是reference、声明之类的
    modified = False
    p_list = xml.xpath('//p')
    if len(p_list) > 0:
        last_p = p_list[-1]
        if last_p.text is not None and last_p.text.startswith('(') and last_p.text.endswith(')'):
            last_p.getparent().remove(last_p)
            modified = True
    return xml, modified


def time_person_email_phonenum_url_discard(xml):
    pat_list = [
        'published on',
        'published by',
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}-\d{2}-\d{2}',  # YY-MM-DD
        r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},\s\d{4}',
        # Month DD, YYYY
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2},\s\d{4}',  # Mon DD, YYYY
        r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}',  # DD Mon YYYY
        r'@[a-zA-Z0-9_]+([\.-][a-zA-Z0-9_]+)*',  # username: @username, @user.name, @user-name et al.
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
        r'\d{3}-\d{3}-\d{4}',  # phone number
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',  # url
        r'\b\d{1,2}[:.]\d{2}\s*(a|p)?\.?m\.?\b',  # time: 12:30pm, 12:30 pm, 12:30 PM, 12:30PM, 12:30PM, 12.30pm et al.
        r'\b([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?\b',  # time 12:30, 12:30:45, 1:30, 01:30, 01:30:45 et al.
        r'\b([01]?[0-9]|2[0-3])(:[0-5][0-9])?([ap]\.?m\.?)\b',  # 12pm, 12:30pm, 12:30:45pm, 1pm, 01pm, 01:30pm et al.
        r'\b([01]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\.\d{1,6}\b',  # 12:30:45.123, 1:30:45.123, 01:30:45.123456 et al.
    ]
    pat_list = [re.compile(pat) for pat in pat_list]
    modified = False
    leaf_nodes = xml.xpath('main//*[not(*)]')
    for node in leaf_nodes:
        if node.tag == 'lb':  # e.g. <p>Follow@htshowbizfor more<lb/>The author tweets @RohanNaahar</p>
            lb_p = node.getparent()
            p_text = lb_p.text if lb_p.text is not None else ''
            p_tail = lb_p.tail if lb_p.tail is not None else ''
            lb_text = node.text if node.text is not None else ''
            lb_tail = node.tail if node.tail is not None else ''
            text = ' '.join([p_text, p_tail, lb_text, lb_tail])
            node = lb_p
        else:
            text = node.text
            if node.tail is not None:
                node_text = node.text if node.text is not None else ''
                text = node_text + ' ' + node.tail
        if text is not None and \
                len(text.split(' ')) < 10:
            delete = False
            for pat in pat_list:
                if pat.search(text) is not None:
                    delete = True
                    break
            if delete:
                parent = node.getparent()
                if parent is not None:  # 有一个p节点有两个lb节点的情况
                    parent.remove(node)
                modified = True
    return xml, modified


def reference_sponsore_discard(xml):
    delete = False
    if len(xml.xpath('main')) == 0:
        return xml, delete
    nodes = list(xml.xpath('main')[0].iter())
    num_ele = len(nodes)
    last_one_fourth = num_ele - num_ele // 4 if num_ele > 20 else 0
    # 如果检测到短的ele有 reference 的字样，并且位置处于整个文章的后四分之一，直接它后面所有的内容删除
    # for elem in xml.xpath(f'main/*[position() > {last_one_fourth}]'):
    for elem in nodes[last_one_fourth:]:
        if elem.text is not None and \
                len(elem.text.split(' ')) < 10 and 'reference' in elem.text.lower():
            delete = True
        if delete or (elem.text is not None and 'sponsore' in elem.text.lower()):
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)
    return xml, delete


def square_brackets_discard(xml):
    modified = False
    # 1. 先识别出所有的 [x] xxxx 的参考文献格式
    p2del = []
    pat_1 = re.compile('\[\d{1,2}\]\s.*(?<=\.)')  # 以方括号开头，后跟1-2个数字，然后是一个空格和任意数量的字符，直到句点为止。
    for p in xml.xpath('//p'):
        if p.text is not None and pat_1.match(p.text):  # 不能用 search, 会误删正文
            p2del.append(p)

    # 2. 整体看识别出来的p是否在文章结尾，如果在结尾，就认定为参考文献
    if len(p2del) > 0 and p2del[-1] in xml.xpath(f'//p[position() > last() - {5}]'):  # 最后一个匹配上的p在结尾才会删除
        # 先删除结尾的参考文献
        for p in p2del:
            p.getparent().remove(p)
            modified = True
        # 删除正文的引用
        pat_2 = re.compile(b'\s\[\d{1,2}\]')
        xml_str = etree.tostring(xml)
        xml_str = pat_2.sub(b'', xml_str)
        xml = etree.fromstring(xml_str)
    return xml, modified


def readme_discard(xml):
    """Discard the paragraph with sentence that are not finished.
    Parag level.
    """
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None

        if text is None:
            continue
        # print(text)
        if not Is_finished(text):
            # print(text)
            # print(text)
            try:
                paragraph.getparent().remove(paragraph)
                modified = True
            except:
                continue
    return xml, modified


def fulltext_discard(xml):
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        text = paragraph.text
        if text is None or text == '':
            continue
        text_new = remove_incomplete_sentence(text)
        # print(text)
        if text_new != text:
            modified = True
            paragraph.text = text_new
            if text_new != '':
                break
        elif text_new == text and text_new != '':
            break

        text = paragraph.tail
        if text is None or text == '':
            continue
        text_new = remove_incomplete_sentence(text)
        # print(text)
        if text_new != text:
            modified = True
            paragraph.tail = text_new
            if text_new != '':
                break
        elif text_new == text and text_new != '':
            break

    paragraphs = xml.xpath('//p | //lb')
    for i in range(len(paragraphs) - 1, -1, -1):
        paragraph = paragraphs[i]
        # print(dir(paragraph))
        text = paragraph.tail
        # print(text)
        if text is None or text == '':
            continue
        text_new = remove_incomplete_sentence_last(text)
        # print(text)
        if text_new != text:
            modified = True
            paragraph.tail = text_new
            if text_new != '':
                break
        elif text_new == text and text_new != '':
            break

        text = paragraph.text
        # print(text)
        if text is None or text == '':
            continue
        text_new = remove_incomplete_sentence_last(text)
        # print(text)
        if text_new != text:
            modified = True
            paragraph.text = text_new
            if text_new != '':
                break
        elif text_new == text and text_new != '':
            break

    return xml, modified


def url_discard(xml):
    modified = False
    try:
        xml_str = etree.tostring(
            xml, encoding='utf-8').decode()
        # print(all_lines)
        xml_str = xml_str.replace("&gt;", ">").replace("&lt;", "<")
        xml_new = etree.fromstring(xml_str)
    except Exception as e:
        # print(e)
        xml_new = xml

    url_pattern = r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]|[a-zA-Z0-9][-a-zA-Z0-9]+([@.][a-zA-Z0-9][-a-zA-Z0-9/]+){2,}/?'
    for paragraph in xml_new.xpath('//*'):
        tail = paragraph.tail
        if tail:
            text = paragraph.text + tail if paragraph.text else tail
        else:
            text = paragraph.text
        if text:
            paragraph.text = re.sub(url_pattern, '', text.strip(), flags=re.I)
            if paragraph.text != text:
                modified = True

    return xml, modified


def image_caption_discard(xml, length_threshold=8):
    modified = False
    # main = xml.xpath("main")[0]
    # print()
    to_remove = []
    elements = xml.xpath("//*")
    for index, element in enumerate(elements):
        if element.tag == "graphic":
            url = element.get("src")
            # print(url)
            if url:
                if os.path.splitext(url)[-1].lower() in [".html"]:
                    # print("discard graphic",url)
                    to_remove.append(index)
            else:
                # print("graphic element has no src. discard graphic")
                to_remove.append(index)
            for i in reversed(range(0, index)):
                if elements[i].tag == "graphic":
                    break
                paragraph = elements[i]
                if paragraph.text is not None and paragraph.tail is not None:
                    text = ''.join([paragraph.text, paragraph.tail])
                elif paragraph.text is not None:
                    text = paragraph.text
                elif paragraph.tail is not None:
                    text = paragraph.tail
                else:
                    text = None
                if text:
                    if paragraph.tag != "head":
                        text = text.lower()
                        matches = re.findall(r"\w+", text)
                        if len(matches) < length_threshold:
                            for word in IMAGE_CAPTION_WORDS:
                                if word in text:
                                    to_remove.append(i)
                                    break
                    break
            for i in range(index + 1, len(elements)):
                if elements[i].tag == "graphic":
                    break
                paragraph = elements[i]
                text = paragraph.text
                if text:
                    if paragraph.tag != "head":
                        if paragraph.text is not None and paragraph.tail is not None:
                            text = ''.join([paragraph.text, paragraph.tail])
                        elif paragraph.text is not None:
                            text = paragraph.text
                        elif paragraph.tail is not None:
                            text = paragraph.tail
                        else:
                            text = None
                        text = text.lower()
                        matches = re.findall(r"\w+", text)
                        if len(matches) < length_threshold:
                            for word in IMAGE_CAPTION_WORDS:
                                if word in text:
                                    to_remove.append(i)

                                    break
                    break
    if len(to_remove) > 0:
        to_remove = set(to_remove)
        for index in to_remove:
            paragraph = elements[index]
            # print(f"discard {paragraph.tag}")

            parent = paragraph.getparent()
            if parent is not None:
                # if paragraph.text:
                # print(f"discard {paragraph.text}")
                parent.remove(paragraph)
                # print(f"remove {paragraph}")

                modified = True

    return xml, modified


def advertisement_discard(xml):
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None

        if text is None:
            continue
        if text.lower() == "advertisement":
            paragraph.getparent().remove(paragraph)
            modified = True
            # print("discard advertisement")

    domain = ["www.merriam-webster", "/dictionary"]
    url = xml.get("source")
    if url:
        to_remove = []
        if domain[0] in url and domain[1] in url:
            elements = xml.xpath("//*")
            for index, paragraph in enumerate(elements):
                if paragraph.text is not None and paragraph.tail is not None:
                    text = ''.join([paragraph.text, paragraph.tail])
                elif paragraph.text is not None:
                    text = paragraph.text
                elif paragraph.tail is not None:
                    text = paragraph.tail
                else:
                    text = None
                if text:
                    text = text.lower()
                    for key_word in MERRIAM_WEBSTER_DICTIONARY:
                        if key_word.endswith("*"):
                            key_word = key_word[:-1]
                        if key_word in text:
                            to_remove.append(index)
            if len(to_remove) > 0:
                index = min(to_remove)
                # print("discard www.merriam-webster")
                for i in range(index, len(elements)):

                    paragraph = elements[i]
                    if paragraph.tag not in ["comments", "main"]:
                        parent = paragraph.getparent()
                        if parent is not None:
                            parent.remove(paragraph)
                        modified = True

    return xml, modified


def re_short_long_paragraphs(xml):
    """Remove paragraphs with short words."""
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None
        if text is not None:
            text = text.strip()
            words = len(text.split())
            if words <= 2:
                parent = paragraph.getparent()
                parent.remove(paragraph)
                modified = True
    return xml, modified


def check_paragraph_lengths(xml):
    """delete doc where all paragraphs are evenly spaced and short"""
    max_paragraph_length = 0
    total_words = 0
    total_paragraphs = 0

    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None
        if text is not None:
            text = text.strip()
            words = len(text.split())
            if words > max_paragraph_length:
                max_paragraph_length = words
            total_words += words
            total_paragraphs += 1

    if max_paragraph_length < 35 and (total_paragraphs == 0 or (total_words / total_paragraphs) < 12):
        return False
    else:
        return True


def multi_lang_word_tokenize(text, language):
    if language == 'en':
        return word_tokenize(text)
    elif language == 'zh':
        return list(jieba.cut(text))
    else:
        # print('Current Version only support English and Chinese, It will be used in English mode.')
        return word_tokenize(text)


def words_discard(text, words_count_range=[50, 100000], avg_word_len_range=[3, 10], words=None, return_words=False,
                  language='en'):
    """
    1. The number of words in [50, 100,000] and average word length in [3, 10] (discard);
    """
    if words is None:
        words = multi_lang_word_tokenize(text, language)
    words_count = len(words)
    if words_count == 0:
        return False if not return_words else False, words
    avg_word_len = sum(len(word) for word in words) / words_count
    if words_count_range[0] <= words_count <= words_count_range[1] and avg_word_len_range[0] <= avg_word_len <= \
            avg_word_len_range[1]:
        return True if not return_words else True, words
    else:
        return False if not return_words else False, words



def char_discard(text, char_threshold=0.8, words=None, language='en'):
    """
    4. 80% of words in a document contain at least one alphabetic character (discard);
    """
    if words is None:
        words = multi_lang_word_tokenize(text, language)
    words_count = len(words)
    if words_count == 0:
        return False
    alphabetic_char_count = 0
    for word in words:
        if any(char.isalpha() for char in word):
            alphabetic_char_count += 1
    if (alphabetic_char_count / words_count) > char_threshold:
        return True
    else:
        return False


def stop_word_discard(text, stop_word_threshold=2, words=None, language='en'):
    """
    5. Apply a "stop word" filter, to remove documents that do not contain at least two of the following English words: the, be, to, of, and, that, have, with (discard).
    """
    if words is None:
        words = multi_lang_word_tokenize(text, language)
    stop_word_count = 0
    for word in words:
        if word in ['the', 'is', 'are', 'am', 'to', 'of', 'and', 'that', 'have', 'with']:
            stop_word_count += 1
    if stop_word_count >= stop_word_threshold:
        return True
    else:
        return False


def document_porn_discard(text, thld=0.02):
    return compute_flagged_word_ratio(text) <= thld


def img_txt_ratio_discard(text, image_count, min_image_count=2, min_text_image_ratio=50):
    """Discard doc with many images and little text.
    Doc level.
    """
    lines = text.split('\n')
    text_count = 0
    result_lines = []
    for line in lines:
        text_count += len(line.split())
        result_lines.append(line)
    text_image_ratio = text_count / max(1, image_count)
    if image_count > min_image_count and text_image_ratio < min_text_image_ratio:
        return False
    else:
        return True


## Paragraph level filters


def numerical_discard(xml, numerical_threshold=0.8, immutable=False):
    """
    2. If it is only composed of numerical characters (discard);
    Eligible paragraphs have graphic retention in the top and bottom 10 lines of the paragraph.
    """
    all_lines = etree.tostring(
        xml, encoding='utf-8').decode('utf-8').split('<')
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None
        if text is None:
            continue
        numerical_char_count = sum(char.isdigit() for char in text)
        if len(text) == 0 or (numerical_char_count / len(text)) > numerical_threshold:
            # immunize paragraphs with graphic retention
            if immutable:
                try:
                    line_number = paragraph.sourceline
                    start_line = max(0, line_number - 10)
                    end_line = min(len(all_lines), line_number + 10)
                    content = ''.join(all_lines[start_line:end_line])
                    if 'graphic' not in content:
                        paragraph.getparent().remove(paragraph)
                        modified = True
                except:
                    continue
            else:
                paragraph.getparent().remove(paragraph)
                modified = True
    return xml, modified


def social_media_counter_discard(xml):
    """
    3. If it is a social media counter (e.g. 3 likes) (discard);
    """
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None
        if text is None:
            continue
        for social_media_tag in ['likes', 'followers', 'subscribes', 'share', 'comments', 'comments:', 'dislikes',
                                 'comment', 'comment.']:
            if text.lower().endswith(social_media_tag):
                try:
                    paragraph.getparent().remove(paragraph)
                    modified = True
                except:
                    continue
    return xml, modified


def one_word_discard(xml, immutable=False):
    """
    4. If it only contains one word (discard);
    """
    all_lines = etree.tostring(
        xml, encoding='utf-8').decode('utf-8').split('<')
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None
        if text is None:
            continue
        if len(text.split()) == 1:
            # immunize paragraphs with graphic retention
            if immutable:
                try:
                    line_number = paragraph.sourceline
                    # if line_number is None:
                    #     print('NOTE!! line_number is None, this iteration will be continued.')
                    start_line = max(0, line_number - 10)
                    end_line = min(len(all_lines), line_number + 10)
                    content = ''.join(all_lines[start_line:end_line])
                    if 'graphic' not in content:
                        paragraph.getparent().remove(paragraph)
                        modified = True
                except:
                    continue
            else:
                paragraph.getparent().remove(paragraph)
                modified = True
    return xml, modified


def short_discard(xml, short_threshold=10):
    """
    5. If it is short (≤ 10 words) and matches a pattern (edit):
        At the beginning of the line (e.g. sign-in);
        At the end of the line (e.g. Read more...);
        Anywhere in the line (e.g. items in cart).
    """
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None
        if text is None:
            continue
        if len(text.split()) <= short_threshold:
            for edit_pattern in ['sign-in', 'sign in', 'sign up', 'read more', 'item in cart']:
                if edit_pattern in text.lower():
                    try:
                        paragraph.getparent().remove(paragraph)
                        modified = True
                    except:
                        continue
    return xml, modified


def porn_discard(xml, thld=0.02):
    """
    """
    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None
        if text is None:
            continue
        if compute_flagged_word_ratio(text) >= thld:
            try:
                paragraph.getparent().remove(paragraph)
                modified = True
            except:
                continue
    return xml, modified


def comments_discard(xml):
    """Delete all comments.
    Parag level.
    """
    modified = False
    for comments in xml.xpath('//comments'):
        for element in comments.getchildren():
            comments.remove(element)
            modified = True
    return xml, modified


def header_footer_discard(xml, max_text_len=10):
    """Delete all lines containing email, phone number, date, or header/footer keywords.
    Parag level.
    """
    email_pattern = r'\S+@\S+\.\S+'
    phone_number_pattern = r'[\(]?\d{3}[\)]?[-\s.]?\d{3}[-\s.]?\d{4}'
    date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b'

    modified = False
    for paragraph in xml.xpath('//p | //lb'):
        if paragraph.text is not None and paragraph.tail is not None:
            text = ''.join([paragraph.text, paragraph.tail])
        elif paragraph.text is not None:
            text = paragraph.text
        elif paragraph.tail is not None:
            text = paragraph.tail
        else:
            text = None
        if text is None:
            continue

        contains_keyword = any(keyword in text for keyword in HEADER_FOOTER_KEYWORDS)
        if contains_keyword or len(text) < max_text_len:
            paragraph.getparent().remove(paragraph)
            modified = True
            continue

        line_contains_email = re.search(email_pattern, text)
        line_contains_phone = re.search(phone_number_pattern, text)
        line_contains_date = re.search(date_pattern, text)
        if (line_contains_email or line_contains_phone or line_contains_date) \
                and (len(text) < max_text_len):
            paragraph.getparent().remove(paragraph)
            modified = True

    return xml, modified


def newlines_discard(xml, end_chars=".!?:;)'\""):
    '''Discard the lines that end with some special characters.
    Parag level.
    '''
    modified = False
    for element in xml.xpath('//*'):
        if element.text is None:
            continue
        last = 'a'
        result = []
        for char in element.text:
            if char not in string.whitespace:
                last = char
            if char == '\n' and last not in end_chars:
                continue
            result.append(char)
        result = ''.join(result)
        if result != element.text:
            modified = True
            element.text = result

        if element.tail is None:
            continue
        last = 'a'
        result = []
        for char in element.tail:
            if char not in string.whitespace:
                last = char
            if char == '\n' and last not in end_chars:
                continue
            result.append(char)
        result = ''.join(result)
        if result != element.tail:
            modified = True
            element.tail = result

    return xml, modified


def heads_discard(xml):
    '''Discard if there is no content between this head and the next head.
    Parag level.
    '''
    modified = False
    headers_to_discard = []

    lst_rend, lst_head = None, None
    for element in xml.xpath('//*'):
        if element.tag == 'head':
            rend = element.get('rend')
            if rend is not None and lst_rend is not None and rend == lst_rend:
                headers_to_discard.append(lst_head)
            lst_rend = rend
            lst_head = element
        else:
            lst_rend, lst_head = None, None

    for header in headers_to_discard:
        modified = True
        header.getparent().remove(header)

    return xml, modified


def video_field_discard(xml):
    """Discard the `[Video]` field in the xml file.
    Parag level.
    """
    modified = False
    xml_str = etree.tostring(xml)
    if b'[Video]' in xml_str:
        xml_str = xml_str.replace(b'[Video]', b'')
        modified = True
        xml = etree.fromstring(xml_str)
    return xml, modified

