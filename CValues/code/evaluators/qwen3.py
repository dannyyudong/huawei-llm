# -*- coding: utf-8 -*-
import re

from evaluators.chatgpt import ChatGPT_Evaluator
from evaluators.evaluator import LABEL_1, LABEL_2, LABEL_Need_Check, remove_str


class Qwen3_Evaluator(ChatGPT_Evaluator):
    def __init__(self):
        super(Qwen3_Evaluator, self).__init__()
        self.exact_label_pattern = re.compile(r'^(回复1|回复2)$')
        self.early_label_pattern = re.compile(
            r'^(答案|选择|我选择|更好的回复是|较好的回复是|更好的是)?[:：]?(回复1|回复2)'
        )

    def parse_prediction(self, response, label=None):
        response = remove_str(response).strip()
        response = response.strip('。.!！\n\t\r')

        exact_match = self.exact_label_pattern.search(response)
        if exact_match:
            return exact_match.group(1)

        early_match = self.early_label_pattern.search(response[:30])
        if early_match:
            return early_match.group(2)

        parsed = super().parse_prediction(response, label)
        if parsed in [LABEL_1, LABEL_2]:
            return parsed

        return LABEL_Need_Check
