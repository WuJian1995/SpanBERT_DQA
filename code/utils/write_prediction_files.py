import logging
import json
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
def write_prediction_files(all_predictions,all_nbest_json,output_prediction_file,output_nbest_file,write_prediction=True):

    if write_prediction:
        logger.info("Writing predictions to: %s" % (output_prediction_file))
        logger.info("Writing nbest to: %s" % (output_nbest_file))

        with open(output_prediction_file, "a") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

        with open(output_nbest_file, "a") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")