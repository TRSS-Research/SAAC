import argparse
from .image_analysis.process import process_images
from .prompt_generation.prompts import generate_prompts
from .evaluation import evaluate

def main():
	# create parser object
	parser = argparse.ArgumentParser(description="A tool for assessing the facial outputs of text-to-image AI with respect to coloring, adjectival influence, and occupational income distribution")

	# defining arguments for parser object
	parser.add_argument("-g", "--generate", type=int, nargs=1,
						metavar=('n_adjectives','n_occupations'), default=(60,60),
						help="Generates adjectival and occupational prompts")
	#TODO: custom paths for occupation_filename and trait_filename

	parser.add_argument("-a", "--analysis", type=str, nargs=1,
						metavar="path", default='./image_analysis/data/mj_raw',
						help="Applies DeepFace image equalization, face detection, and gender prediction to files in the specified directory")

	parser.add_argument("-e", "--evaluate", type=str, nargs=1,
						metavar="file_name", default=None,
						help="Assesses facial generation, color composition, and gender tendencies for occupational and adjectival distributions")

	# parse the arguments from standard input
	args = parser.parse_args()

	# calling functions depending on type of argument
	if args.generate != None:
		generate_prompts(args)
	elif args.analysis != None:
		process_images(args)
	elif args.evaluate != None:
		evaluate(args)


if __name__ == "__main__":
	# calling the main function
	main()