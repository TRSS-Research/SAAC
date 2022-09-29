import json
import os
import jmespath as jp
from task import generate_prompts as bigbench_prompts
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sid_analyzer = SentimentIntensityAnalyzer()


def loadBold(folderpath='./bold-main/wikipedia'):
	output = []
	if os.path.exists(folderpath) and os.path.isdir(folderpath):
		for node in os.listdir(folderpath):
			fp = os.path.join(folderpath, node)
			if os.path.isfile(fp):
				with open(fp, 'r') as f:
					data = json.load(f)

					for outerkey in data.keys():
						tags = []
						# race,religion,gender,character,job = t
						if 'gender' in node:
							tags.append(('', '', outerkey, '', ''))
						elif 'race' in node:
							tags.append((outerkey, '', '', '', ''))
						elif 'religio' in node:
							tags.append(('', outerkey, '', '', ''))
						else:
							tags.append(('', '', '', '', outerkey))
						for subkey in data[outerkey]:
							output.append(('bold', data[outerkey][subkey][0].strip().replace('\"', ''), tags[0]))
	return output


def load_kaggle_mj(folderpath='./kaggle_mj'):
	output = []
	if os.path.exists(folderpath) and os.path.isdir(folderpath):
		for node in os.listdir(folderpath):
			fp = os.path.join(folderpath, node)
			if os.path.isfile(fp):
				with open(fp, 'r') as f:
					data = json.load(f)
					p = jp.search('messages[:][:].content', data)
					p = p[0][0].replace('\"', '')
					if '**' in p and '<https://' not in p:
						p = p[int(p.find('**') + 2):p.rfind('**')]
						output.append(('mj', p, ('')))
	return output


def datasets_to_csv(filename='prompts.csv'):
	p = loadBold('./bold-main/prompts')
	# p.extend(load_kaggle_mj())
	p.extend(bigbench_prompts())
	with open(filename, 'w', encoding='utf8') as f:
		f.write('source,prompt,race,religion,gender,bias,other_tag,\n')
		for s, prmpt, t in p:
			if s == 'bigbench' or s == 'bold':
				print(t, len(t))
				ra, re, g, c, j = t
				f.write(f'{s},\"{prmpt}\",\"{ra}\",\"{re}\",\"{g}\",\"{c}\",\"{j}\",\n')
			else:
				f.write(f'{s},\"{prmpt}\",\"\",\"\",\"\",\"\",\"{t}\",\n')


def imagefile_to_dataframe(directory='./gender'):
	directory = os.path.normpath(directory)
	filenames = dict()
	if os.path.isdir(directory):
		for node in os.listdir(directory):
			fp = os.path.join(directory, node)
			if os.path.isdir(fp):
				filenames[node] = list()
			for f in os.listdir(fp):
				impath = os.path.join(fp, f)
				if os.path.isfile(impath):
					if '.png' in os.path.splitext(f)[0]:
						elements = os.path.splitext(f)[0].split('_')
					else:
						elements = f.split('_')
					#print(elements)
					user = elements.pop(0)
					elements = [e for e in elements if
								not re.match('[A-z-\d]+(\.png)', e) and not re.match('(\d){1}', e) and e != 'photorealistic']
					filenames[node].append((os.path.abspath(impath), elements))
	return filenames


def prompts_by_gender(filenames):
	prompts = dict([(k, list()) for k in fn.keys()])
	hypo_ind = 0
	for key in fn:
		# print(key)
		tokens = dict()
		for f in fn[key]:  # filepath, list of prompt words
			for t in f[1]:
				if t not in tokens:
					tokens[t] = 0
				tokens[t] += 1

			# sort by frequency
			ftokens = sorted(tokens, key=lambda k: tokens[k], reverse=True)

			sent_tokens = dict()
			for l in ['pos', 'neg', 'neu']:
				sent_tokens[l] = set()
			# assess sentiment of unique tokens
			for k in ftokens:
				scores = sid_analyzer.polarity_scores(k)
				if scores['compound'] < 0:
					sent_tokens['neg'].add(k)
				elif scores['compound'] > 0:
					sent_tokens['pos'].add(k)
				else:
					sent_tokens['neu'].add(k)
			print(key, sent_tokens)
			toremove = []
			for i in range(len(f[1])):
				# print(f[1])
				if f[1][i] == 'photorealistic':
					toremove.append(i)
					continue
				# modify/condense
				if f[1][i] == 'person':
					f[1][i] = '{person}'
				if f[1][i] in sent_tokens['pos'] or f[1][i] in sent_tokens['neg']:
					f[1][i] = '{bias}'
			for j in toremove:
				f[1].pop(j)
			prompts[key].append(f"{' '.join(f[1])}, photorealistic --s 625")
		# hypo_ind += 1 if hypo_ind < len(hypos) else 0

		print(key, sent_tokens)
	return prompts


if __name__ == '__main__':
	# datasets_to_csv('bold_prompts.csv')
	fn = imagefile_to_dataframe('./midjourney_zs')
	#print(fn)
	hypos = list()
	# with open('person_hyponyms.txt', 'r') as f:
	# 	for l in f.readlines():
	# 		hypos.append(l.strip())
	# prompts = prompts_by_gender(fn)
	prompts = []
	for f in fn['test']:
		prompts.append("/imagine prompt:"+" ".join(f[1])+', photorealistic --s 625')

	print(prompts)
	with open('zs_prompts.csv', 'w') as f:
		f.write('\"key\",\"prompt\"\n')
		for k in prompts:
			f.write(f"\"{k}\",\"{k}\"\n")
