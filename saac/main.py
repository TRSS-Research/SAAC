import argparse
import os

from saac.image_analysis.process import process_images
from saac.prompt_generation.prompts import generate_prompts
from saac.evaluation import evaluate,eval_utils
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
def main():
	# create parser object
	parser = argparse.ArgumentParser(description="A tool for assessing the facial outputs of text-to-image AI with respect to coloring, adjectival influence, and occupational income distribution")

	# defining arguments for parser object
	parser.add_argument("-g", "--generate", type=str, nargs="?",
						metavar = "path",
						const = os.path.join(os.getcwd(),'generated_prompts.csv'),
						help="Generates adjectival and occupational prompts saving to the specified filename (defaults to generated_prompts.csv).")
	parser.add_argument("--num_adj",type=int, nargs="?", const=60,default=60, help="number of adjectives to sample positive, negative, and neutral for prompt generation")
	parser.add_argument("--num_occ", type=int, nargs="?", const=60,default=60, help="Number of occupations to sample high/med/low salaries for prompt_generation")
	parser.add_argument("-a", "--analysis", type=str, nargs="?",
						metavar="path",
						const=os.path.join(MAIN_DIR, 'image_analysis', 'data', 'mj_raw'),
						help="Applies DeepFace image equalization, face detection, and gender prediction to files in the specified directory")
	parser.add_argument("--analysis_output", type=str, nargs="?", metavar="path",
						const=os.path.join(os.getcwd(), 'processed.csv'),
						help="CSV filepath to output results of image analysis")
	parser.add_argument("-e", "--evaluate", type=str, nargs="?",
						metavar="path",
						const=os.path.join(MAIN_DIR,'evaluation','data','processed'),
						help="Assesses facial generation, color composition, and gender tendencies for occupational and adjectival distributions. "
							 "Expects a directory containing an Occupation_Results.csv and TDA_Results.csv or requires input of --analysis_file")
	parser.add_argument("--analysis_file",type=str,nargs="?",metavar="path",
						const=os.path.join(MAIN_DIR, 'image_analysis', 'data', 'processed.csv'),
						help="Specifies the csv containing face detection, gender detection, and skin color tuples per imagefile."
						)
	parser.add_argument("-f",'--force',type=bool,default=False,nargs="?",const=True,
						help="Whether or not to force re-computation/processing")
	# print(parser)
	# parse the arguments from standard input
	args = parser.parse_args()

	print(args)
	# calling functions depending on type of argument
	if args.generate is not None:
		num_adj,num_occ = args.num_adj,args.num_occ
		generate_prompts(sampledims=(num_adj,num_occ),force=args.force,save_path=args.generate)
	if args.analysis is not None:
		print('Processing ',args.analysis,' to ',args.analysis_output)
		process_images(raw_images_dir=args.analysis,force=args.force,output_file=args.analysis_output)
	if args.evaluate is not None:
		if args.analysis_file is not None and os.path.exists(args.analysis_file):
			print('Processing ')
			eval_utils.process_analysis(args.analysis_file,savepath=args.evaluate)
		evaluate(processed_filedir=args.evaluate,force=args.force)


if __name__ == "__main__":
	# calling the main function
	main()