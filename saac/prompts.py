# TODO Separate out to mj individual section in repo
# TODO Add remaining mj params + style banks + artist banks
# https://github.com/willwulfken/MidJourney-Styles-and-Keywords-Reference
# https://github.com/ymgenesis/Midjourney-Photography-Resource
# TODO Create funcs + section for dale + stable dif
# https://github.com/jina-ai/discoart


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
