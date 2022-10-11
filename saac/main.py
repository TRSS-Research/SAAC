import argparse
import os.path

from saac.image_analysis.process import process_images
from saac.prompt_generation.prompts import generate_prompts
from saac.evaluation import evaluate

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
def main():
	# create parser object
	parser = argparse.ArgumentParser(description="A tool for assessing the facial outputs of text-to-image AI with respect to coloring, adjectival influence, and occupational income distribution")

	# defining arguments for parser object
	parser.add_argument("-g", "--generate", type=int, nargs="?",
						const=(60,60),
						help="Generates adjectival and occupational prompts")

	parser.add_argument("-a", "--analysis", type=str, nargs="?",
						metavar="path",
						const=os.path.join(MAIN_DIR,'image_analysis','data','mj_raw'),
						help="Applies DeepFace image equalization, face detection, and gender prediction to files in the specified directory")

	parser.add_argument("-e", "--evaluate", type=str, nargs="?",
						metavar="path",
						const=os.path.join(MAIN_DIR,'evaluation','data','processed'),
						help="Assesses facial generation, color composition, and gender tendencies for occupational and adjectival distributions")

	# parse the arguments from standard input
	args = parser.parse_args()

	print(args)
	# calling functions depending on type of argument
	if args.generate is not None:
		generate_prompts(sampledims=args.generate)
	if args.analysis is not None:
		process_images(raw_images_dir=args.analysis)
	if args.evaluate is not None:
		evaluate(processed_filedir=args.evaluate)


if __name__ == "__main__":
	# calling the main function
	main()