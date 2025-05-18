from collections import defaultdict
from typing import Any
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from transformers import Pipeline
from transformers.pipelines import pipeline
from typing import Callable
import re

default_qa_template = "Based on the image, answer the following question by strictly selecting only one option from the given choices.\nQuestion: {question}\nAnswer:"

def is_answer_match(ans: str, should: str) -> bool:
    ans = ans.lower().strip()
    should = should.lower().strip()

    option_part = should.split(")")[0] + ")"  # "(b)"
    desc_part = should.split(") ")[1]  # "7 years"
    option_letter = option_part[1]  # "b"

    # 构造正则表达式，匹配：
    # 1. 整个 should
    # 2. 仅选项部分（如 "(b)"）
    # 3. 仅描述部分（如 "7 years"）
    # 4. 仅选项字母（如 "b"），且必须是独立字母
    pattern = rf"^({re.escape(should)}|{re.escape(option_part)}|{re.escape(desc_part)}|\b{option_letter}\b)$"
    return bool(re.fullmatch(pattern, ans))

class VQAScorer:
    def __init__(self, set_curr_score: Callable[[int], None], template: str=default_qa_template) -> None:
        self.set_curr_score = set_curr_score
        self.template = template

    def calc_score(
            self, vqa_pipeline: Pipeline, images: torch.Tensor, prompts: tuple[str], metadata: tuple[Any]
    ):
        batch_size = len(images)
        scores = [0.0] * len(images)
        to_pil = ToPILImage()

        vqa_samples = []

        for i, image in enumerate(images):
            all_qa: list[dict[str, str]] = (
                metadata[i]["qa"]["relation"] + metadata[i]["qa"]["attribute"]
            )
            if isinstance(image, torch.Tensor):
                pil_image = to_pil(image.to(torch.float))
            for each_qa in all_qa:
                vqa_samples.append(
                    (
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": pil_image,
                                    },
                                    {
                                        "type": "text",
                                        "text": self.template.format(question=each_qa["question"]),
                                    },
                                ],
                            }
                        ],
                        each_qa["answer"],
                        len(all_qa),
                        i,
                    )
                )

        # calc reward in batch
        for i in range(0, len(vqa_samples), batch_size):
            q_with_image, answers, qa_lens, img_indices = zip(*vqa_samples[i: i + batch_size])
            responses = vqa_pipeline(text=q_with_image, max_new_tokens=512, return_full_text=False)  # type: ignore

            for response, answer, qa_len, img_idx in zip(responses, answers, qa_lens, img_indices):
                generated_answer = response[0]["generated_text"]
                # print(generated_answer)
                scores[img_idx] += (1 / qa_len) if is_answer_match(generated_answer, answer) else 0
            # print(scores)

        return np.array(scores), None  # type: ignore
