import os
import logging
import joblib


def save_to_disk(object, path, filename):
    """

    :param object:
    :param path:
    :param filename:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info("Saving object")

    outdir = os.path.dirname(path)
    if not os.path.exists(os.path.join(outdir, path)):
        os.makedirs(os.path.join(outdir, path))

    logger.info("Saving object of type:{}".format(type(object)))

    joblib.dump(object, os.path.join(path, filename))
    logger.info("Object saved: {}".format(
        os.path.join(path, filename)))
