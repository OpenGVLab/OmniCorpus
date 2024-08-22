import re
import string
from lxml import etree
from nltk import word_tokenize

IMAGE_CAPTION_WORDS = ["download", "load", "getty", "image", "photo", "source"]

MERRIAM_WEBSTER_DICTIONARY = ["word of the day", "test Your *", "keep scrolling for more", "take the quiz"]



main_special_characters = string.punctuation + string.digits + string.whitespace
other_special_characters = (
    "    　    ￼’“”–ー一▬…✦�­£​•€«»°·═"
    "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
    "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
    "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
    "」﴾》"
)
SPECIAL_CHARACTERS = set(main_special_characters + other_special_characters)

STOPWORDS = [
    "a",
    "a.k.a",
    "aboard",
    "about",
    "above",
    "abt",
    "accord",
    "according",
    "across",
    "after",
    "against",
    "ago",
    "aground",
    "ahead",
    "aka",
    "ala",
    "albeit",
    "all",
    "along",
    "alongside",
    "although",
    "am",
    "amid",
    "amidst",
    "among",
    "amongst",
    "amoung",
    "an",
    "and",
    "and/or",
    "another",
    "any",
    "any1",
    "anybody",
    "anyone",
    "anything",
    "are",
    "around",
    "as",
    "aside",
    "astride",
    "at",
    "atop",
    "away",
    "b",
    "b/c",
    "b/t",
    "back",
    "base",
    "based",
    "bc",
    "be",
    "because",
    "been",
    "before",
    "behind",
    "being",
    "below",
    "beneath",
    "beside",
    "besides",
    "between",
    "beyond",
    "board",
    "both",
    "btwn",
    "but",
    "by",
    "can",
    "cause",
    "circa",
    "cos",
    "could",
    "coz",
    "cus",
    "depend",
    "depending",
    "despite",
    "did",
    "do",
    "does",
    "down",
    "due",
    "during",
    "each",
    "either",
    "else",
    "even",
    "ever",
    "every",
    "everybody",
    "everyone",
    "everything",
    "except",
    "for",
    "forth",
    "from",
    "get",
    "gets",
    "getting",
    "give",
    "given",
    "got",
    "had",
    "half",
    "has",
    "hav",
    "have",
    "having",
    "he",
    "her",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "i",
    "i'd",
    "if",
    "in",
    "include",
    "including",
    "inside",
    "instead",
    "into",
    "is",
    "it",
    "it's",
    "its",
    "itself",
    "lest",
    "like",
    "made",
    "many",
    "may",
    "me",
    "might",
    "mine",
    "minus",
    "most",
    "much",
    "must",
    "my",
    "myself",
    "nary",
    "near",
    "nearby",
    "neither",
    "next",
    "nigh",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "notwithstanding",
    "of",
    "off",
    "on",
    "onboard",
    "once",
    "one",
    "ones",
    "oneself",
    "only",
    "onto",
    "opposite",
    "or",
    "other",
    "others",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "outside",
    "over",
    "overt",
    "own",
    "past",
    "per",
    "plus",
    "prior",
    "quite",
    "rather",
    "re",
    "regard",
    "regarding",
    "regardless",
    "round",
    "s/he",
    "save",
    "self",
    "shall",
    "she",
    "should",
    "side",
    "since",
    "so",
    "some",
    "somebody",
    "someone",
    "something",
    "such",
    "sure",
    "teh",
    "than",
    "thanks",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "they're",
    "thier",
    "this",
    "tho",
    "those",
    "thou",
    "though",
    "through",
    "throughout",
    "thru",
    "thy",
    "til",
    "till",
    "to",
    "together",
    "too",
    "toward",
    "towards",
    "u",
    "under",
    "underneath",
    "unless",
    "unlike",
    "until",
    "unto",
    "up",
    "upon",
    "ur",
    "us",
    "use",
    "versus",
    "via",
    "vs",
    "vs.",
    "w/",
    "w/o",
    "w/out",
    "was",
    "we",
    "were",
    "what",
    "whatever",
    "whatnot",
    "when",
    "whenever",
    "where",
    "whereas",
    "wherever",
    "whether",
    "which",
    "while",
    "whilst",
    "whither",
    "who",
    "who's",
    "whoever",
    "whom",
    "whomever",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "wo",
    "worth",
    "would",
    "wud",
    "y'all",
    "ya",
    "yet",
    "yo",
    "you",
    "you're",
    "your",
    "youre",
    "yours",
    "yourself",
    "yourselves",
]
STOPWORDS = set(STOPWORDS)

FLAGGED_WORDS = [
    "anal",
    "bareback",
    "bbw",
    "bdsm",
    "blowjob",
    "blowjobs",
    "brazzers",
    "bukkake",
    "camgirl",
    "camwhore",
    "cocksucking",
    "creampie",
    "cuckold",
    "cum",
    "cumming",
    "cums",
    "cumshot",
    "cumshots",
    "cumslut",
    "cunnilingus",
    "deepthroat",
    "deepthroating",
    "dildo",
    "dildos",
    "dogging",
    "doggystyle",
    "dominatrix",
    "erotic",
    "fellatio",
    "femdom",
    "fingering",
    "fisting",
    "footjob",
    "gangbang",
    "handjob",
    "hentai",
    "horney",
    "horniest",
    "horny",
    "jism",
    "jizz",
    "masterbating",
    "masturbate",
    "masturbating",
    "masturbation",
    "milf",
    "orgies",
    "orgy",
    "pegging",
    "porn",
    "pornhub",
    "porno",
    "pornos",
    "pornstar",
    "pornstars",
    "redtube",
    "rimming",
    "slutty",
    "squirting",
    "strapon",
    "threesome",
    "vibrator",
    "xhamster",
    "xnxx",
    "xvideos",
    "xxx",
    "youporn",
]
FLAGGED_WORDS = set(FLAGGED_WORDS)


HEADER_FOOTER_KEYWORDS = [
    "We appreciate your help to end bad business websites",
    "Not Your Average WordPress Web Design Agency",
    "Subscribe to receive exclusive content and notifications",
    "Services",
    "Any questions?",
    "Downloadable software products",
    "Some health and personal care items",
    "Please contact us through our Contact Us page to open a return ticket.",
    "Top image",
    "Like Liked",
    "Looking for",
    "How do I login to my zoom account",
    "Watch on YouTube",
    "Contact Information",
    "Read More",
    "Showing 1–9 of 7361 results",
    "Call Us",
    "There are no reviews yet",
    "Related Fellow",
    "Want More",
    "More information",
    "Login and Registration",
    "Google+",
    "YouTube",
    "Subscribe",
    "Available for you in",
    "Sign in",
    "view details",
    "ID #",
    "Thank you so much for reading"
]


def strip(text, strip_characters):
    """Way faster than text.strip(strip_characters)
    since strip_characters is a set instead of a str,
    and it contains a lot of elements (all the emojis)."""
    if not text:
        return text
    beg_ind = 0
    end_ind = len(text)
    for i in range(len(text)):
        if text[i] in strip_characters:
            beg_ind += 1
        else:
            break
    for i in range(1, len(text) + 1):
        if text[-i] in strip_characters:
            end_ind -= 1
        else:
            break
    text_stripped = text[beg_ind:end_ind]
    return text_stripped


def remove_empty_el_from_list(list_):
    return [el for el in list_ if el]


def split_on_whitespace(
        text,
        new_line=False,
        tab=False,
):
    """This method also removes concatenated spaces."""
    sep = [" "] + new_line * ["\n"] + tab * ["\t"]
    sep = "|".join(sep)
    split_text = re.split(sep, text)
    split_text = remove_empty_el_from_list(split_text)
    return split_text


def get_words_from_text(text, lower_case=True, strip_words=True, strip_characters=None):
    """Get words from a text. Non reversible since the text
    is split on multiple characters, words are stripped of
    special characters and characters are converted to lower case.
    Useful to compute ratios, like the stopword ratio."""
    if strip_words and strip_characters is None:
        raise ValueError("strip_characters must be provided if strip_words is True.")
    words = split_on_whitespace(text=text, new_line=True, tab=True)
    if lower_case:
        words = [word.lower() for word in words]
    if strip_words:
        words = [strip(word, strip_characters) for word in words]
        words = remove_empty_el_from_list(words)
    return words


def compute_flagged_word_ratio(text, strip_characters=SPECIAL_CHARACTERS, flagged_words=FLAGGED_WORDS):
    words = get_words_from_text(
        text=text, lower_case=True, strip_words=True, strip_characters=strip_characters
    )
    if not words:
        return 0
    flagged_word_ratio = len([word for word in words if word in flagged_words]) / len(words)
    return flagged_word_ratio


def Iscontatined(words, black_words):
    for i in range(len(black_words) - 1, len(words)):
        flag = words[i].lower() == black_words[-1]
        for j in range(1, len(black_words)):
            flag = flag and (words[i - j].lower() == black_words[-1 - j])
            if not flag:
                break
        if flag:
            return True
    return False


def Is_finished(text,
                black_list=['read more', 'more words', 'read next', 'learn more', 'see this article', 'click here',
                            'continue reading', "Read Full Article", "See More", "Click to View Details",
                            "Read the Rest", "Expand", "Show More", "Continue Browsing", "Read Next Section",
                            "View Full Content", "Read On", "See Remaining", "Continue Watching", "Explore More"]):
    for black_word in black_list:
        black_words = black_word.split(' ')
        words = word_tokenize(text)
        if Iscontatined(words, black_words) and check_last_sentence(text, black_words):
            return False
    return True


def remove_comments_content_from_xml(xml_string):
    root = etree.fromstring(xml_string)
    flag = False
    # 查找<comments>标签
    comments_element = root.find('comments')

    if comments_element is not None:
        # 删除<comments>标签及其内容
        if len(list(comments_element)) > 0:
            flag = True
        root.remove(comments_element)

    modified_xml = etree.tostring(root, encoding='utf-8').decode('utf-8')
    return modified_xml


def check_last_sentence(text, black_words):
    # 切分文本为句子
    black_word = black_words[0] + ' ' + black_words[1]
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    # 获取最后一句话
    last_sentence = sentences[-1] if sentences else ""
    # print(last_sentence)
    # 判断是否包含"Read more"（不区分大小写）
    words = word_tokenize(last_sentence)
    # 检查当前元素是否是'more'，前一个元素是否是'read'
    if Iscontatined(words, black_words):
        # print(words)
        if last_sentence[-1] not in ['.', '!', '?']:
            # 判断"Read more"前后的条件是否满足
            before_read_more = last_sentence[:last_sentence.lower().rfind(black_word)]
            after_read_more = last_sentence[last_sentence.lower().rfind(black_word) + len(black_word):]
            # if re.search(r'[a-zA-Z0-9]', before_read_more) and not re.search(r'[a-zA-Z0-9]', after_read_more):
            if not re.search(r'[a-zA-Z0-9]', after_read_more):
                return True
        if len(sentences) == 1:
            return True
    return False


def remove_incomplete_sentence_last(text_old):
    sentences = re.split(r'((?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)(?:"|\')?\s)', text_old)
    # print(sentences)
    start_id = len(sentences) - 1
    # print(sentences)
    while start_id >= 0:
        if sentences[start_id] != '' and sentences[start_id][-1] not in ['.', '!', '?', "”", "\""]:
            start_id -= 1
        else:

            return ''.join(sentences[:start_id + 1])
    return ''


def remove_incomplete_sentence(text_old):
    sentences = re.split(r'((?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)(?:"|\')?\s)', text_old)
    # print(sentences)
    start_id = 0
    while start_id < len(sentences):
        if sentences[start_id] != '' and sentences[start_id][-1] not in ['.', '!', '?', "”", "\""]:
            start_id += 1
        else:

            return ''.join(sentences[start_id:])
    return ''