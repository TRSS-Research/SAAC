import json
import os
import jmespath as jp
from task import generate_prompts
def loadBold(folderpath='./bold-main/wikipedia'):
	output = []
	if os.path.exists(folderpath) and os.path.isdir(folderpath):
		for node in os.listdir(folderpath):
			fp = os.path.join(folderpath,node)
			if os.path.isfile(fp):
				with open(fp,'r') as f:
					data = json.load(f)

					for outerkey in data.keys():
						tags = []
						# race,religion,gender,character,job = t
						if 'gender' in node:
							tags.append(('','',outerkey,'',''))
						elif 'race' in node:
							tags.append((outerkey,'','','',''))
						elif 'religio' in node:
							tags.append(('',outerkey,'','',''))
						else:
							tags.append(('','','','',outerkey))
						for subkey in data[outerkey]:
							output.append(('bold',data[outerkey][subkey][0].strip().replace('\"',''),tags[0]))
	return output

def load_kaggle_mj(folderpath='./kaggle_mj'):
	output = []
	if os.path.exists(folderpath) and os.path.isdir(folderpath):
		for node in os.listdir(folderpath):
			fp = os.path.join(folderpath,node)
			if os.path.isfile(fp):
				with open(fp,'r') as f:
					data = json.load(f)
					p = jp.search('messages[:][:].content',data)
					p = p[0][0].replace('\"', '')
					if '**' in p and '<https://' not in p:
						p = p[int(p.find('**')+2):p.rfind('**')]
						output.append(('mj',p,('')))
	return output

def datasets_to_csv(filename='prompts.csv'):
	p = loadBold('./bold-main/prompts')
	# p.extend(load_kaggle_mj())
	p.extend(generate_prompts())
	with open(filename,'w',encoding='utf8') as f:
		f.write('source,prompt,race,religion,gender,bias,other_tag,\n')
		for s,prmpt,t in p:
			if s=='bigbench' or s=='bold':
				print(t,len(t))
				ra,re,g,c,j = t
				f.write(f'{s},\"{prmpt}\",\"{ra}\",\"{re}\",\"{g}\",\"{c}\",\"{j}\",\n')
			else:
				f.write(f'{s},\"{prmpt}\",\"\",\"\",\"\",\"\",\"{t}\",\n')

if __name__=='__main__':
	datasets_to_csv('bold_prompts.csv')