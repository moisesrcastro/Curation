from typing import List


class TextProcessing:
    """
    Stateless text preprocessing utilities.
    Responsible only for text normalization and cleaning.
    """

    def __init__(self):
        pass


    def clean_html(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        return (
            text.replace("<p>", "")
            .replace("</p>", "")
            .replace("<br>", " ")
            .replace("&nbsp;", " ")
            .replace('<span class="ql-cursor">﻿</span>', "")
            .strip()
        )

    def clean_html_batch(self, texts: List[str]) -> List[str]:
        return [self.clean_html(t) for t in texts]


    def normalize_text(self, text: str) -> str:
        """
        Normalize text for downstream semantic comparison.
        """
        text = self.clean_html(text)
        return " ".join(text.split()).lower()

    def normalize_text_batch(self, texts: List[str]) -> List[str]:
        return [self.normalize_text(t) for t in texts]
