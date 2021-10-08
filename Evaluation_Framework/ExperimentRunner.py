import numpy as np
import os
from os.path import join
import json
import random
import logging
from timeit import default_timer as timer
from multiprocessing import Pool
from utils.EvaluationScore import EvaluationScore
from utils.UserDataReader import UserDataReader
from Models.Model import Model



class ExperimentRunner:

    def __init__(self, num_examples, num_test, users_path, data_path, min_range, max_index):
        self.num_examples = num_examples
        self.num_test = num_test
        self.users_path = users_path
        self.data_path = data_path
        self.min_range = min_range
        self.max_range = max_index
        
        # create a user data reader object
        self.udr = UserDataReader(self.users_path, self.data_path, self.min_range, self.max_range)
        # create evaluation scorer
        self.eval_scorer = EvaluationScore()

        # read the data used for testing
        self.ground_truth = self.udr.read_summaries_list()
        self.intent_doc = np.array(self.udr.read_intent_doc())
        self.sudocu_data = np.array(self.udr.read_example_summaries_sudocu())
        self.intent_list = self.udr.read_intents()

        # list containing the range of examples to split into examples/test data
        self.ex_range = [i for i in range(0, (num_examples+num_test))]


    ###
    #    Helper function to return the topic score and length constraints
    #        used by SuDocu
    ### 
    def evaluate_example(self, model, target_doc, gt_summary, example_summaries, intent, output_file_path, ex_num, pool_num, trial_num, processed_ctr):
        
        # probability of saving off this file randomly
        save_prob = random.uniform(0, 1)

        # time used to track how long it takes to create the summary
        eval_start_time = timer()

        # predict the summary
        predicted_summary = model.get_predicted_summary(target_doc, example_summaries, processed_ctr)

        # eval end time
        eval_end_time = timer()

        # get the Rouge scores of the summaries
        scores = self.eval_scorer.compareScore(predicted_summary, gt_summary)

        # ~ 10% chance of being saved off
        if save_prob < 0.1:
            with open(join(output_file_path, "txts/", "ex_{}_pool_{}_trial_{}_summaries.txt".format(ex_num, pool_num, trial_num)), "w") as outfile:
                    outfile.write("Document:\n")
                    outfile.write(target_doc)
                    outfile.write("\n\nIntent:\n")
                    outfile.write(intent)
                    outfile.write("\n\Predicted Summary:\n")
                    outfile.write(predicted_summary)
                    outfile.write("\n\nGT:\n")
                    outfile.write(gt_summary)
                    outfile.write("\n\nRouge-1, Rouge-2, Rouge-L: [p, r, f1, f2]:\n")
                    for rouge_results in scores:
                        outfile.write(str(rouge_results) + "\n")
                    outfile.close()
        
        return (scores, (eval_end_time - eval_start_time))


    ###
    #    Helper function to return the topic score and length constraints
    #        used by SuDocu
    ### 
    def model_get_evaluation(self, model, trial_num, output_file_path, multi_processing=True):
        
        processed_ctr = 0          # keep track of the number of processed examples
        all_example_scores = []    # list to hold the all of the rouges scores for every example
        all_runtimes = []          # list to hold all of the runtimes for every example
        all_scores_by_intent = {}  # dictionary to hold scores by intent

        # iterate over every intent to initialize the by intent
        #     scores dictionary
        for intent in self.intent_list:
            all_scores_by_intent[intent] = []

        # iterate over every intent, ground truth, and example summary set tuples
        for i, (intent, ground_truth, example_summaries) in enumerate(zip(self.intent_doc, self.ground_truth, self.sudocu_data)):
            # start a timer to track how long it takes to complete a single, complete
            #    evaluation loop
            start_total = timer()
            
            # periodically print results
            if processed_ctr % 10 == 0:
                logging.info("processed: %d examples" % (processed_ctr))
             
            # get the text of the current intent
            intent_text = list(intent.keys())[0]
            
            # get the documents that where summarized in this example for this intent
            documents = np.array(list(intent.values())[0])

            # get a random training subset
            train_split = random.sample(range(0, len(example_summaries)), self.num_examples)
                
            # list to hold the test indices for this combination
            test_indices = []
            
            # extract the test indices
            for index in self.ex_range:
                if index not in train_split:
                    test_indices.append(index)
                    
            # get the set of example summaries
            example_set = example_summaries[train_split]
                
            # get the set of test documents
            test_docs = documents[np.array(test_indices)]
            
            # get the ground_truth summaries for the test set
            test_gt = np.array(ground_truth)[np.array(test_indices)]
                
            # variable to hold the json of all example summaries
            # input_example_summary = json.dumps(example_set.tolist())

            # array to hold the results for all three test summaries
            scores_per_batch = []
            
            # runtimes per batch
            runtimes_per_batch = []

            # log that we are solving the test examples
            if processed_ctr % 10 == 0:
                    logging.info("Evaluating Test Examples") 
            
            # variable to keep track of the pool number this test file is being summarized with
            pool_num = 0

            # conditional for using multi_processing
            if (multi_processing == True):
                # list to hold arguments for pool workers
                args_list = []
                    
                # (self, model, target_doc, gt_summary, example_summaries, intent, output_file_path, ex_num, pool_num, trial_num, processed_ctr):

                for doc, gt_summ in zip(test_docs, test_gt):
                    temp_tup = (model, doc, gt_summ, example_set.tolist(), intent_text, output_file_path, i, pool_num, trial_num, processed_ctr)
                    args_list.append(temp_tup)
                    pool_num += 1   

                # run the processing pool to solve for the test documents
                res = None
                with Pool() as pool:
                    res = pool.starmap(self.evaluate_example, args_list)
                
                # iterate over the results and add them to the appropriate lists
                for rouge_scores in res:
                    scores_per_batch.append(rouge_scores[0])
                    runtimes_per_batch.append(rouge_scores[1])
                # end of using multi-processing loop
            else:
                for doc, gt_summ in zip(test_docs, test_gt):
                    score, runtime = self.evaluate_example(model, doc, gt_summ, example_set.tolist(), intent_text, output_file_path, i, pool_num, trial_num, processed_ctr)
                    pool_num += 1
                    scores_per_batch.append(score)
                    runtimes_per_batch.append(runtime)

            # end_time of evaluation loop
            end_total = timer()
            
            # end of the conditional solve statements
            scores_per_batch = np.array(scores_per_batch)
            runtimes_per_batch = np.array(runtimes_per_batch)

            # some quick quality assurance tests
            assert scores_per_batch.shape[0] == self.num_test and scores_per_batch.shape[-1] == 4, "ExperimentRunner->model_get_evaluation: avg score per batch incorrect shape"
            assert runtimes_per_batch.shape[0] == self.num_test, "ExperimentRunner->model_get_evaluation: runtimes per batch incorrect shape"

            # preiodically print update
            if processed_ctr % 10 == 0:
                logging.info('total elapsed time for example: %.4f seconds\n\n' % (end_total - start_total))
           
            # increment the number of processed examples
            processed_ctr += 1
            
            # add this batch score to the appropriate intent
            all_scores_by_intent[intent_text].append(scores_per_batch.tolist())
            
            # append the p/r scores to the list containing all p/r scores for all examples
            all_example_scores.append(scores_per_batch)
            all_runtimes.append(runtimes_per_batch)

        # end of all examples for-loop
        return (np.array(all_example_scores).squeeze(), np.array(all_runtimes).squeeze(), all_scores_by_intent)




    ###
    #    Function called in experiment scripts to kick off model analysis 
    ### 
    def get_model_analysis(self, model, num_trials, display_results, model_name, exp_folder=None, multi_processing=True):
        # get the current working directory 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        
        # make the overall results folder if it does not exist
        if not os.path.exists(join(dir_path, "Results/")):
            os.makedirs(join(dir_path, "Results/"))

        # variable to hold the output filepath for this experiment
        output_file_path = join(dir_path, "Results/")

        # if we are saving these results into an experiments folder
        #     add it to the filepath and make it if it doesn't exist
        if (exp_folder != None):
            output_file_path = join(output_file_path, exp_folder)

            # make the experiment folder if it doesn't already exist
            if not os.path.exists(output_file_path):
                os.makedirs(output_file_path)
        
        # add the specific model we are evaluating to the path
        output_file_path = join(output_file_path, model_name)

        # make the output-model directory if it does not already exist
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

        # lastly, create the txts/ folder to hold the saved off summaries
        if not os.path.exists(join(output_file_path, "txts/")):
            os.makedirs(join(output_file_path, "txts/"))

        # directory up-keep complete, run experiment
        logging.info("Directory Construction Complete, begining experiments")

        # loop over the number of trials
        for i in range(num_trials):
            ex_scores, ex_runtimes, intent_results = self.model_get_evaluation(model, i, output_file_path, multi_processing)

            # save off the results
            # save total scores as NPZ indexed by model
            np.savez(join(output_file_path, ("all_scores_range_%s_%s_trail_%d" % (str(self.min_range), str(self.max_range), i))), scores=ex_scores)
            
            # save total runtimes as NPZ indexed by model
            np.savez(join(output_file_path, ("all_runtimes_%s_%s_trail_%d" % (str(self.min_range), str(self.max_range), i))), scores=ex_runtimes)
            
            # save the by-intent sorted results
            with open(join(output_file_path, ("all_scores_range_by_intent_%s_%s_trail_%d.txt" % (str(self.min_range), str(self.max_range), i))), 'w') as outfile:
                json.dump(intent_results, outfile)
                outfile.close()

            # display trail results if appropriate
            if (display_results):
                avg_results = np.mean(ex_scores, axis=0)
                logging.info("*********************** trial num: %d *************************" % i)
                logging.info("\nAverage p, r, f1, and f2 scores")
                logging.info("Rouge-1, Rouge-2, Rouge-L: [p, r, f1, f2]:")
                for rouge_results in avg_results:
                    logging.info(rouge_results)
                logging.info("\n\n")


