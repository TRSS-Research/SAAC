# TODO Separate out to mj individual section in repo
# TODO Add remaining mj params + style banks + artist banks
# https://github.com/willwulfken/MidJourney-Styles-and-Keywords-Reference
# https://github.com/ymgenesis/Midjourney-Photography-Resource
# TODO Create funcs + section for dale + stable dif
# https://github.com/jina-ai/discoart
# %%
import os
from typing import Callable, Optional

from .prompt_utils import score_sentiment, generate_traits, generate_occupations, PROMPT_GENERATION_DATA_DIR
import pandas as pd


def mj_prompt(text,
              photorealistic: bool = True,
              stylized: int = 625):
    stylized = max(stylized, 625)
    stylized = min(stylized, 60000)

    start_arg = "/imagine prompt:"

    style = []
    if photorealistic:
        style.append("photorealistic")

    stylize_param = f" --s {stylized}"

    prompt = start_arg + text + " " + ",".join(style) + stylize_param

    return prompt


def stable_diffusion_prompt(text,
                            photorealistic: bool = True):
    return text + f", photorealistic" if photorealistic else ''


def generate_prompts(output_dir: str = None,
                     sampledims: tuple[int, int] = (60, 60),
                     prompt_wrapper: Optional[Callable] = mj_prompt,
                     occupation_filename:Optional[str]=None,
                     trait_filename:Optional[str]=None
                     ) -> pd.DataFrame:
    if output_dir is None:
        output_dir = os.path.join(PROMPT_GENERATION_DATA_DIR, 'processed')
    trait_n, occ_n = sampledims

    if occupation_filename is None or len(occupation_filename) < 1:
        occ_n = min(12 * 5, occ_n)
    if trait_filename is None or len(trait_filename) < 1:
        trait_n = min(500 * 5, trait_n)

    trait_samples = generate_traits(nsamples=int(trait_n / 5), filepath=trait_filename)
    occ_samples = generate_occupations(nsamples=int(occ_n / 5), filepath=occupation_filename)
    prompts_df = pd.concat([trait_samples, occ_samples])
    prompts_dfv = score_sentiment(prompts_df, 'prompt', verbose=True)
    prompts_dfv['prompt'] = prompts_dfv['prompt'].apply(prompt_wrapper)
    prompts_dfv.to_csv(os.path.join(output_dir, 'generated_prompts.csv'), index=False)
    return prompts_dfv


if __name__ == '__main__':
    generate_prompts(prompt_wrapper=stable_diffusion_prompt)
