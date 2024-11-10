import logging
import os

def set_logger(save_path):
    logger = logging.getLogger("Train Anomaly")
    logger.setLevel(logging.INFO)  # Set the level for the logger itself

    # Avoid duplicate handlers if this function is called multiple times
    if not logger.hasHandlers():
        # Create a console handler and set the level to info
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a file handler and set the level to info
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_handler = logging.FileHandler(os.path.join(save_path, "log.txt"))
        file_handler.setLevel(logging.INFO)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add both handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

def main():
    logger = set_logger(save_path='./')
    logger.info('Start of the program')
    logger.info("Finished")

if __name__ == '__main__':
    main()