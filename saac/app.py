#initial fork from https://huggingface.co/spaces/Manjushri/SD-2.1-CPU/raw/main/app.py

import gradio as gr
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import collections
import numpy as np
import pandas as pd
import io
from prompt_generation.prompts import generate_prompts,generate_occupations,generate_traits
from prompt_generation.prompt_utils import score_prompt
from evaluation.eval_utils import generate_countplot,process_analysis,generate_histplot
from datasets import load_dataset
from diffusers import DiffusionPipeline, PNDMScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
STABLE_MODELS = ["runwayml/stable-diffusion-v1-5", "Midjourney"]

# TODO Make model selectable input
scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", prediction_type="v_prediction")
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=scheduler)
pipe = pipe.to(device)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

#
# AUTHORS NOTE:
# Being absolutely explicit: the genders and ethnicities listed here
# are reductive. This is done not to erase but make analysis here
# simpler, tractable and easier to understand
#
GENDERS = ["male", "female"]
ETHNICITIES = ["black", "white", "asian"]
LOOKS = list(generate_traits()['tag'])#["beautiful", "stunning", "handsome", "ugly", "plain", "repulsive", "arrogant", "trustworthy"]
JOBS = list(generate_occupations()['tag'])#["CEO", "doctor", "nurse", "cashier", "janitor", "engineer", "pilot", "dentist", "leader"]
RENDERPREFIX = "a high quality photo of a"

def echoToken(token):
    res = getMostSimilar(tokenizer, text_encoder, token)
    return ",".join(res)

def getEmbeddingForToken(tokenizer, token):
    token_ids = tokenizer.encode(token)[1:-1]
    if len(token_ids) != 1:
        print(len(token_ids))
        raise
    token_id = token_ids[0]
    return token_id, text_encoder.get_input_embeddings().weight.data[token_id].unsqueeze(0)

def getMostSimilar(tokenizer, text_encoder, token, numResults=50):
    internal_embs = text_encoder.text_model.embeddings.token_embedding.weight
    tID, tok = getEmbeddingForToken(tokenizer, token)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = cos(internal_embs.to("cpu").to(torch.float32), tok.to("cpu").to(torch.float32))
    sorted_scores, sorted_ids = torch.sort(scores, descending=True)
    best_ids = sorted_ids[0:numResults].detach().numpy()
    best_scores = sorted_scores[0:numResults].detach().numpy()

    res = []
    for best_id, best_score in zip(best_ids, best_scores):
        #res.append((tokenizer.decode(best_id), best_score))
        res.append("[" + tokenizer.decode(best_id) + "," + str(best_score) + "]")
    return res[1:]

def computeTermSimilarity(tokenizer, text_encoder, termA, termB):
    inputs = tokenizer([termA, termB], padding=True, return_tensors="pt").to("cpu")
    outputs = text_encoder(**inputs)
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    val = cos(outputs.pooler_output[0], outputs.pooler_output[1]).item()                   
    return float(val)

def computeJob(tokenizer, text_encoder, job):
    res = {}
    neutralPrompt = " ".join([RENDERPREFIX, job])
    titleText = neutralPrompt
    for gender in GENDERS:
        for ethnicity in ETHNICITIES:
            prompt = " ".join([RENDERPREFIX, ethnicity, gender, job])
            val = computeTermSimilarity(tokenizer, text_encoder, prompt, neutralPrompt)
            res[prompt] = val
            
    return titleText, sorted(res.items(), reverse=True)

def computeLook(tokenizer, text_encoder, look):
    res = {}
    titleText = " ".join([RENDERPREFIX, 
                          look,
                          "[",
                          "|".join(GENDERS),
                          "]"])

    for gender in GENDERS:
        neutralPromptGender = " ".join([RENDERPREFIX, look, gender])
        for ethnicity in ETHNICITIES:
            prompt = " ".join([RENDERPREFIX, look, ethnicity, gender])
            val = computeTermSimilarity(tokenizer, text_encoder, prompt, neutralPromptGender)
            res[prompt] = val
    
    return titleText, sorted(res.items(), reverse=True)

# via https://stackoverflow.com/questions/57316491/how-to-convert-matplotlib-figure-to-pil-image-object-without-saving-image
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def computePlot(title, results, scaleXAxis=True):
    x = list(map(lambda x:x[0], results))
    y = list(map(lambda x:x[1], results))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    y_pos = np.arange(len(x))

    hbars = ax.barh(y_pos, y, left=0, align='center')
    ax.set_yticks(y_pos, labels=x)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Cosine similarity - take care to note compressed X-axis')
    ax.set_title('Similarity to "' + title + '"')

    # Label with specially formatted floats
    ax.bar_label(hbars, fmt='%.3f')
    minR = np.min(y)
    maxR = np.max(y)
    diffR = maxR-minR

    if scaleXAxis:
        ax.set_xlim(left=minR-0.1*diffR, right=maxR+0.1*diffR)
    else:
        ax.set_xlim(left=0.0, right=1.0)
    plt.tight_layout()
    plt.close()
    return fig2img(fig)

def computeJobBias(job):
    title, results = computeJob(tokenizer, text_encoder, job)
    return computePlot(title, results)

def computeLookBias(look):
    title, results = computeLook(tokenizer, text_encoder, look)
    return computePlot(title, results)
def trait_graph(trait,hist=True):
    tda_res, occ_result = process_analysis()
    fig = None
    if not hist:
        fig = generate_countplot(tda_res, 'tda_sentiment_val', 'gender_detected_val',
                       title='Gender Count by Trait Sentiment',
                       xlabel='Trait Sentiment',
                       ylabel='Count',
                       legend_title='Gender')
    else:
        df = tda_res
        df['tda_sentiment_val'] = pd.Categorical(df['tda_sentiment_val'],
                                             ['very negative', 'negative', 'neutral', 'positive', 'very positive'])
        fig = generate_histplot(tda_res, 'tda_sentiment_val', 'gender_detected_val',
                      title='Gender Distribution by Trait Sentiment',
                      xlabel='Trait Sentiment',
                      ylabel='Count', )

    return fig2img(fig)
def occ_graph(occ):
    tda_res, occ_result = process_analysis()
    fig = generate_histplot(occ_result, 'a_median', 'gender_detected_val',
                   title='Gender Distribution by Median Annual Salary',
                   xlabel= 'Median Annual Salary',
                   ylabel= 'Count',)
    return fig2img(fig)

if __name__=='__main__':
    disclaimerString = ""

    jobInterface = gr.Interface(fn=occ_graph,
                                 inputs=[gr.Dropdown(JOBS, label="occupation")],
                                 outputs='image',
                                 description="Referencing a specific profession comes loaded with associations of gender and ethnicity."
                                             " Text to image models provide an opportunity to explicitly specify an underrepresented group, but first we must understand our default behavior.",
                                title="How occupation affects txt2img gender and skin color representation",
                                 article = "To view how mentioning a particular occupation affects the gender and skin colors in faces of text to image generators, select a job."
                                           " Promotional materials, advertising, and even criminal sketches which do not explicitly specify a gender or ethnicity term will tend towards the displayed distributions.")

    affectInterface = gr.Interface(fn=trait_graph,
                                   inputs=[gr.Dropdown(LOOKS, label="trait")],
                                   outputs='image',
                                   description="Certain adjectives can reinforce harmful stereotypes associated with gender roles and ethnic backgrounds."
                                               "Text to image models provide an opportunity to understand how prompting a particular human expression could be triggering,"
                                               " or why an uncommon combination might provide important examples to minorities without default representation.",
                                   title="How word sentiment affects txt2img gender and skin color representation",
                                   article = "To view how characterizing a person with a positive, negative, or neutral term influences the gender and skin color composition of AI-generated faces, select a direction.")

    jobInterfaceManual = gr.Interface(fn=score_prompt,
                                      inputs=[gr.inputs.Textbox()],
                                      outputs='text',
                                      description="Analyze prompt",
                                      title="Understand which prompts require further engineering to represent equally genders and skin colors",
                                      article = "Try modifying a trait or occupational prompt to produce a result in the minority representation!")


    toolInterface = gr.Interface(fn=lambda t: trait_graph(t,hist=False),inputs=[gr.Dropdown(STABLE_MODELS,label="text-to-image model")],outputs='image',
                                title="How different models fare in gender and skin color representation across a variety of prompts",
                                 description="The training set, vocabulary, pre and post processing of generative AI tools doesn't treat everyone equally. "
                                             "Within a 95% margin of statistical error, the following tests expose bias in gender and skin color.",
                                 article="To learn more about this process, <a href=\"http://github.com/trss/facia.git\"/> Visit the repo</a>"
                                 )

    gr.TabbedInterface(
        [jobInterface, affectInterface, jobInterfaceManual,toolInterface],
        ["Occupational Bias", "Adjectival Bias", "Prompt analysis",'FACIA model auditing'],
        title = "Text-to-Image Bias Explorer"
    ).launch(share=True)
