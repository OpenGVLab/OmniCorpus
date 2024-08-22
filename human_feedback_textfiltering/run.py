import io
import json
from lxml import etree
from meta import xmltodoc
from myxml import xmltotxt
from mytraf import mydetermine_returnstring
from text_filter.filters import (words_discard,
                                 char_discard, stop_word_discard, document_porn_discard,
                                 img_txt_ratio_discard, uppercase_discard, numerical_discard,
                                 social_media_counter_discard, one_word_discard, short_discard,
                                 porn_discard, comments_discard, header_footer_discard,
                                 newlines_discard, heads_discard, underlines_split,
                                 video_field_discard,
                                 readme_discard, fulltext_discard, url_discard, image_caption_discard,
                                 advertisement_discard, re_short_long_paragraphs,
                                 check_paragraph_lengths,
                                 tooshort_discard, aberrant_item_discard, cite_discard,
                                 social_media_discard, phonenum_author_time_discard,
                                 filter_download_links, filter_source_references)


def filter_single_xml_en(xml_str):
    doc = xmltodoc(xml_str)
    xml = etree.fromstring(xml_str)
    _, num_images = mydetermine_returnstring(
        doc, output_format='txt', include_formatting=False, tei_validation=False)

    xml, _ = newlines_discard(xml)
    xml, _ = underlines_split(xml)
    xml, _ = video_field_discard(xml)
    xml, _ = fulltext_discard(xml)

    # Paragraph Level Filtering
    xml, _ = filter_download_links(xml)
    xml, _ = filter_source_references(xml)
    xml, _ = uppercase_discard(xml, uppercase_threshold=0.8, immutable=True)
    xml, _ = numerical_discard(xml, numerical_threshold=0.8, immutable=True)
    xml, _ = social_media_counter_discard(xml)
    xml, _ = one_word_discard(xml, immutable=True)
    xml, _ = short_discard(xml)
    xml, _ = porn_discard(xml)
    xml, _ = comments_discard(xml)
    xml, _ = header_footer_discard(xml)
    xml, _ = heads_discard(xml)
    xml, _ = readme_discard(xml)
    xml, _ = url_discard(xml)
    xml, _ = image_caption_discard(xml)
    xml, _ = advertisement_discard(xml)
    xml, _ = re_short_long_paragraphs(xml)
    xml, _ = tooshort_discard(xml)
    xml, _ = aberrant_item_discard(xml)
    xml, _ = cite_discard(xml)
    xml, _ = social_media_discard(xml)
    xml, _ = phonenum_author_time_discard(xml)

    pure_txt = xmltotxt(
        xml, include_formatting=False, include_images=False)
    if pure_txt is None or len(pure_txt) == 0:
        return None
    words_keep, words = words_discard(pure_txt, words_count_range=(50, 100000), avg_word_len_range=[3, 10],
                                      return_words=True)
    if not words_keep:
        return None
    if not char_discard(pure_txt, char_threshold=0.8, words=words): return None
    if not stop_word_discard(pure_txt, stop_word_threshold=2, words=words): return None
    if not document_porn_discard(pure_txt, thld=0.02): return None
    if not img_txt_ratio_discard(pure_txt, image_count=num_images, min_image_count=2,
                                 min_text_image_ratio=50): return None
    if not check_paragraph_lengths(xml): return None
    return etree.tostring(xml).decode()

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf8') as file:
        # print(file)
        for line in file:
            # print(line)
            data.append(json.loads(line))
    return data

def main_function(input_file=None, output_file=None):
    datas = read_jsonl_file(file_path=input_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for line_dict in datas:
            xml_txt = line_dict['content']
            res = filter_single_xml_en(xml_txt)
            if res is not None:
                res = filter_single_xml_en(res)
            if res is not None:
                res = filter_single_xml_en(res)
            if res is None:
                continue
            line_dict['content'] = res
            f.write(json.dumps(line_dict, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main_function(input_file="test.jsonl", output_file="output.jsonl")
